"""Mindful Scroll helpers (no Streamlit in here, so everything can be tested on its own).

1. STRICT EXTRACTION  extract_record() forces every new piece of data into one exact JSON schema.
2. STORAGE            log_event() saves validated records to a local SQLite database.
3. PANDAS TOOL        analyze() does the maths; agent_explain() lets the LLM only interpret the result.
4. SIMULATOR          simulate_usage() makes random, schedule-based, teen-like screen time for judges.
5. MACHINE LEARNING   train_usage_model() predicts hours when an app will be used.
"""
import json
import math
import os
import random
import re
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split

APPS = ["Instagram", "TikTok", "Snapchat", "YouTube"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LOCS = ["home", "school", "outside"]
CONFIDENCE = 0.6      # send a habit notification when the model is at least this sure
MIN_ACCURACY = 0.8    # ...and only once its measured accuracy is at least this
FEATURE_NAMES = ["Day of week", "Hour of day", "Is weekend", "Where you just were",
                 "Hours since school starts", "Hours since school ends",
                 "Hours until bedtime", "Hours since wake-up"]

# =====================================================================================
# 1. STRICT JSON SCHEMA
# =====================================================================================
CATEGORIES = ["screen_time", "mindful_minutes", "breathing_break",
              "scroll_skipped", "scroll_opened", "feeling"]
SCHEMA = {
    "type": "object",
    "properties": {
        "date": {"type": "string", "format": "date", "description": "YYYY-MM-DD"},
        "hour": {"type": "integer", "minimum": 0, "maximum": 23},
        "category": {"type": "string", "enum": CATEGORIES},
        "label": {"type": "string", "maxLength": 40, "description": "app name or short tag"},
        "number": {"type": "number", "description": "minutes, a count, or -1..1 for a feeling"},
    },
    "required": ["date", "hour", "category", "label", "number"],
    "additionalProperties": False,
}


def extract_record(raw):
    """Turns raw input (dict or JSON text) into exactly {date, hour, category, label, number}.
    Anything missing, extra, or the wrong type raises ValueError, so nothing unchecked is ever saved."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"not valid JSON: {e}")
    if not isinstance(raw, dict):
        raise ValueError("record must be a JSON object")
    keys = set(SCHEMA["required"])
    if set(raw) != keys:
        raise ValueError(f"fields must be exactly {sorted(keys)}; got {sorted(raw)}")
    d = raw["date"]
    if isinstance(d, (datetime, date)):
        d = d.date().isoformat() if isinstance(d, datetime) else d.isoformat()
    if not isinstance(d, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        raise ValueError("date must be YYYY-MM-DD")
    date.fromisoformat(d)
    hour = raw["hour"]
    if isinstance(hour, bool) or not isinstance(hour, (int, np.integer)) or not 0 <= hour <= 23:
        raise ValueError("hour must be an integer 0-23")
    cat = raw["category"]
    if cat not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    label = raw["label"]
    if not isinstance(label, str):
        raise ValueError("label must be text")
    label = re.sub(r"[^\w \-+.]", "", label).strip()[:40]
    num = raw["number"]
    if isinstance(num, bool) or not isinstance(num, (int, float, np.integer, np.floating)):
        raise ValueError("number must be numeric")
    num = float(num)
    if not math.isfinite(num) or (cat != "feeling" and num < 0) or abs(num) > 100000:
        raise ValueError("number out of range")
    return {"date": d, "hour": int(hour), "category": cat, "label": label, "number": round(num, 3)}


EXTRACT_SYSTEM = (
    "Extract ONE record from the user's message. Reply with JSON only (no prose, no code fences) with "
    "exactly these fields: date (YYYY-MM-DD), hour (integer 0-23), category (one of "
    + ", ".join(CATEGORIES) + "), label (app name or a short tag, may be empty), number (minutes, a count, "
    "or -1 to 1 for a feeling). Use the default date and hour you are given when the message does not say.")


def _regex_extract(text, today, hour_now):
    t = text.lower()
    rec = {"date": today.isoformat(), "hour": hour_now, "category": None, "label": "", "number": 1.0}
    if "yesterday" in t:
        rec["date"] = (today - timedelta(days=1)).isoformat()
    m = re.search(r"\bat (\d{1,2})(?::\d\d)?\s*(am|pm)?", t)
    if m:
        h = int(m.group(1)) % 12 if m.group(2) else int(m.group(1))
        rec["hour"] = (h + 12 if m.group(2) == "pm" else h) % 24
    m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)\b", t)
    if m:
        rec["number"] = float(m.group(1)) * (60 if m.group(2).startswith("h") else 1)
    for app in APPS:
        if app.lower() in t:
            rec["label"] = app
    if rec["label"] or "scroll" in t or "screen" in t:
        rec["category"] = "screen_time"
    elif "breath" in t:
        rec["category"] = "breathing_break"
    elif "skip" in t:
        rec["category"] = "scroll_skipped"
    if rec["category"] is None:
        raise ValueError("could not tell what kind of data this is")
    return rec


def extract_from_text(text, today=None, hour_now=12, client_factory=None, model=None):
    """Free text -> strict record. Uses Claude when a client is given, otherwise a small regex parser.
    Either way the result goes through extract_record(), so the schema is enforced."""
    today = today or date.today()
    if client_factory:
        try:
            client = client_factory()
            msg = client.messages.create(
                model=model, max_tokens=200, system=EXTRACT_SYSTEM,
                messages=[{"role": "user", "content":
                           f"Default date: {today.isoformat()}. Default hour: {hour_now}.\nMessage: {text}"}])
            out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            out = re.sub(r"^```(?:json)?|```$", "", out.strip()).strip()
            return extract_record(out)
        except Exception:
            pass
    return extract_record(_regex_extract(text, today, hour_now))


# =====================================================================================
# 2. LOCAL SQLITE STORAGE
# =====================================================================================
DB_PATH = "mindful.db"


def init_db(path=None):
    global DB_PATH
    if path:
        DB_PATH = path
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT NOT NULL, date TEXT NOT NULL,
            hour INTEGER NOT NULL, category TEXT NOT NULL, label TEXT NOT NULL,
            number REAL NOT NULL, saved_at TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_events_uid ON events(uid, category, date)")


def _insert(c, uid, rec):
    c.execute("INSERT INTO events (uid, date, hour, category, label, number, saved_at) VALUES (?,?,?,?,?,?,?)",
              (uid, rec["date"], rec["hour"], rec["category"], rec["label"], rec["number"],
               datetime.now().isoformat(timespec="seconds")))


def log_event(uid, raw):
    """Step 1 validate (strict schema). Step 2 save. Nothing else happens before validation."""
    rec = extract_record(raw)
    with sqlite3.connect(DB_PATH) as c:
        _insert(c, uid, rec)
    return rec


def store_usage(uid, usage):
    """Replaces this visitor's screen_time rows with a freshly generated week-by-week log."""
    recs = [extract_record({"date": r.date, "hour": int(r.hour), "category": "screen_time",
                            "label": r.app, "number": float(r.minutes)}) for r in usage.itertuples()]
    with sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM events WHERE uid=? AND category='screen_time'", (uid,))
        for rec in recs:
            _insert(c, uid, rec)


def events_df(uid):
    with sqlite3.connect(DB_PATH) as c:
        return pd.read_sql_query("SELECT date, hour, category, label, number, saved_at FROM events "
                                 "WHERE uid=? ORDER BY id", c, params=(uid,))


def recent_events(uid, n=8):
    return events_df(uid).tail(n).drop(columns=["saved_at"]).iloc[::-1].reset_index(drop=True)


def export_csv(uid):
    return events_df(uid).to_csv(index=False)


def delete_user(uid):
    with sqlite3.connect(DB_PATH) as c:
        c.execute("DELETE FROM events WHERE uid=?", (uid,))


# =====================================================================================
# 3. PANDAS ANALYSIS TOOL  (the maths) + LLM (only the interpretation)
# =====================================================================================
def _num(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def analyze(uid, ss, se, bed, wake, days=42):
    """Averages, trends and correlations, computed with pandas from the saved records."""
    df = events_df(uid)
    sc = df[df["category"] == "screen_time"].copy()
    if sc.empty:
        return {"error": "no screen time saved yet"}
    sc["date"] = pd.to_datetime(sc["date"])
    last = sc["date"].max()
    sc = sc[sc["date"] > last - pd.Timedelta(days=int(days))]
    idx = pd.date_range(sc["date"].min(), last)
    daily = sc.groupby("date")["number"].sum().reindex(idx, fill_value=0.0)
    weekend = daily.index.dayofweek >= 5
    n_days = len(daily)

    sc["weekday"] = sc["date"].dt.dayofweek
    school = sc[(sc["weekday"] < 5) & (sc["hour"] >= ss) & (sc["hour"] < se)]
    bed_h = bed if bed > se else 24
    after = sc[(sc["weekday"] < 5) & (sc["hour"] >= se) & (sc["hour"] < bed_h - 2)]
    late = sc[sc["hour"].isin([(bed - 1) % 24, (bed - 2) % 24])]
    table = pd.DataFrame({"total": daily, "weekend": weekend.astype(int)})
    for name, part in (("after_school", after), ("late_night", late), ("school_hours", school)):
        table[name] = part.groupby("date")["number"].sum().reindex(idx, fill_value=0.0)

    slope = np.polyfit(np.arange(n_days), daily.values, 1)[0] * 7 if n_days > 1 else 0.0
    per_app = (sc.groupby("label")["number"].sum() / n_days).sort_values(ascending=False)
    per_hour = (sc.groupby("hour")["number"].sum() / n_days).sort_values(ascending=False)
    corr = table.corr()
    ev = df[df["category"].isin(["scroll_skipped", "scroll_opened", "breathing_break", "mindful_minutes"])]
    skipped = int((ev["category"] == "scroll_skipped").sum())
    opened = int((ev["category"] == "scroll_opened").sum())
    return {
        "days_analyzed": n_days,
        "avg_minutes_per_day": _num(daily.mean()),
        "weekday_avg_minutes": _num(daily[~weekend].mean()),
        "weekend_avg_minutes": _num(daily[weekend].mean()),
        "avg_minutes_per_app_per_day": {k: _num(v) for k, v in per_app.items()},
        "busiest_hours": [{"hour": int(h), "avg_minutes": _num(v)} for h, v in per_hour.head(3).items()],
        "weekly_trend_minutes_per_day_change": _num(slope),
        "correlations": {
            "total_vs_weekend": _num(corr.loc["total", "weekend"], 2),
            "total_vs_after_school": _num(corr.loc["total", "after_school"], 2),
            "total_vs_late_night": _num(corr.loc["total", "late_night"], 2),
            "after_school_vs_late_night": _num(corr.loc["after_school", "late_night"], 2),
        },
        "mindful": {"breathing_breaks": int((ev["category"] == "breathing_break").sum()),
                    "mindful_minutes": _num(ev.loc[ev["category"].isin(["mindful_minutes", "breathing_break"]),
                                                   "number"].sum()),
                    "scrolls_skipped": skipped, "scrolls_opened": opened},
    }


def plain_summary(r):
    """Simple wording used when no AI service is available."""
    if "error" in r:
        return "There is no screen time saved yet, so there is nothing to analyze."
    trend = r["weekly_trend_minutes_per_day_change"]
    way = "up" if trend > 2 else "down" if trend < -2 else "flat"
    top = next(iter(r["avg_minutes_per_app_per_day"]), "none")
    return (f"Over {r['days_analyzed']} days the average is {r['avg_minutes_per_day']:.0f} minutes a day "
            f"({r['weekday_avg_minutes']:.0f} on weekdays, {r['weekend_avg_minutes']:.0f} on weekends). "
            f"{top} takes the most time, and the weekly trend is {way}. "
            f"These are patterns in the data, not certainties.")


TOOL = {
    "name": "analyze_screen_time",
    "description": "Runs pandas on the saved records and returns numbers only: daily averages, weekday vs "
                   "weekend, minutes per app, busiest hours, weekly trend and correlations.",
    "input_schema": {"type": "object",
                     "properties": {"days": {"type": "integer", "minimum": 7, "maximum": 90,
                                             "description": "How many recent days to analyze"}},
                     "required": []},
}
AGENT_SYSTEM = (
    "You are the analysis voice of Mindful Scroll, an app that helps teens use social media mindfully. "
    "ALWAYS call analyze_screen_time first. Never do arithmetic or invent numbers yourself: only explain "
    "the numbers the tool returns, in 3 to 4 short, friendly sentences. Mention one or two patterns, "
    "note correlations are not causes, and do not give medical advice. Plain text.")


def run_tool(name, args, uid, profile):
    if name != "analyze_screen_time":
        return {"error": "unknown tool"}
    days = int(min(90, max(7, (args or {}).get("days", 42))))
    return analyze(uid, profile["ss"], profile["se"], profile["bed"], profile["wake"], days)


def agent_explain(uid, profile, client_factory=None, model=None):
    """The agent calls the pandas tool, then the LLM interprets only the returned numbers.
    Returns (text, report). Without an AI service, a template sentence is used instead."""
    report = run_tool("analyze_screen_time", {}, uid, profile)
    if client_factory is None or "error" in report:
        return plain_summary(report), report
    try:
        client = client_factory()
        msgs = [{"role": "user", "content": "What patterns do you see in my screen time?"}]
        for _ in range(3):
            r = client.messages.create(model=model, max_tokens=500, system=AGENT_SYSTEM,
                                       tools=[TOOL], messages=msgs)
            if r.stop_reason != "tool_use":
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                return (text or plain_summary(report)), report
            msgs.append({"role": "assistant", "content": r.content})
            results = []
            for b in r.content:
                if getattr(b, "type", "") == "tool_use":
                    out = run_tool(b.name, b.input, uid, profile)
                    report = out
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out)})
            msgs.append({"role": "user", "content": results})
    except Exception:
        pass
    return plain_summary(report), report


# =====================================================================================
# 4. SIMULATED SCREEN TIME (random every visit, built from the school schedule)
# =====================================================================================
def is_asleep(h, bed, wake):
    if bed > wake:
        return h >= bed or h < wake
    return bed <= h < wake


def where(d, h, ss, se):
    """Fake location sensor, based on the school schedule."""
    if d >= 5:
        return "outside" if 13 <= h < 18 else "home"
    return "school" if ss <= h < se else "home"


def feat(d, h, ss, se, bed, wake):
    prev = LOCS.index(where(d, (h - 1) % 24, ss, se))
    return [d, h, int(d >= 5), prev, (h - ss) % 24, (h - se) % 24, (bed - h) % 24, (h - wake) % 24]


def simulate_usage(ss, se, bed, wake, seed, days=42, end_iso=None):
    """A believable teen week-after-week: a few habit hours tied to the schedule (after school,
    before bed, lunch...), each with its own chance and session length, plus random one-off use.
    Different seed = different habits. Returns (usage DataFrame, list of planted habit rules)."""
    rng = random.Random(seed)
    end = date.fromisoformat(end_iso) if end_iso else date.today()
    wd = {  # slot name -> (hour, likely apps, minutes range)
        "before school": ((ss - 1) % 24, ["Instagram", "Snapchat"], (8, 25)),
        "lunch": ((ss + se) // 2 % 24, ["Snapchat", "Instagram", "TikTok"], (12, 30)),
        "right after school": (se % 24, ["TikTok", "Snapchat", "Instagram"], (25, 60)),
        "late afternoon": ((se + 1) % 24, ["YouTube", "Snapchat"], (20, 50)),
        "evening": ((bed - 3) % 24, ["TikTok", "YouTube", "Instagram"], (30, 75)),
        "before bed": ((bed - 1) % 24, ["YouTube", "TikTok"], (20, 60)),
    }
    we = {
        "late morning": ((wake + 3) % 24, ["YouTube", "Instagram"], (20, 50)),
        "afternoon": (15, ["Instagram", "TikTok", "Snapchat"], (30, 80)),
        "evening": (20, ["TikTok", "YouTube"], (35, 90)),
        "before bed": ((bed - 1) % 24, ["TikTok", "YouTube"], (20, 60)),
    }

    def pick(slots, must, k):
        names = [n for n, v in slots.items() if not is_asleep(v[0], bed, wake)]
        chosen = [n for n in must if n in names]
        rest = [n for n in names if n not in chosen]
        rng.shuffle(rest)
        chosen += rest[:max(0, k - len(chosen))]
        rules, seen = [], set()
        for n in chosen:
            h, apps, mins = slots[n]
            if h in seen:
                continue
            seen.add(h)
            rules.append({"slot": n, "hour": h, "app": rng.choice(apps),
                          "chance": round(rng.uniform(0.6, 0.92), 2), "minutes": mins})
        return rules

    wk_rules = pick(wd, ["right after school", "before bed"], rng.randint(4, 5))
    we_rules = pick(we, ["evening"], rng.randint(3, 4))
    rows = []
    for i in range(days):
        dt = end - timedelta(days=days - i)
        d = dt.weekday()
        rules = we_rules if d >= 5 else wk_rules
        by_hour = {r["hour"]: r for r in rules}
        for h in range(24):
            if is_asleep(h, bed, wake):
                continue
            r = by_hour.get(h)
            if r and rng.random() < r["chance"]:
                rows.append((dt.isoformat(), d, h, r["app"], int(rng.randint(*r["minutes"]) * 1.25)))
            elif rng.random() < 0.06:
                rows.append((dt.isoformat(), d, h, rng.choice(APPS), rng.randint(5, 20)))
    usage = pd.DataFrame(rows, columns=["date", "weekday", "hour", "app", "minutes"])
    shown = ([dict(r, when="on weekdays") for r in wk_rules] + [dict(r, when="on weekends") for r in we_rules])
    return usage, shown


# =====================================================================================
# 5. MACHINE LEARNING
# =====================================================================================
def build_table(usage, ss, se, bed, wake, end_iso, days=42):
    """One row per awake hour of the tracking period, labeled with the app used or 'none'."""
    end = date.fromisoformat(end_iso)
    used = {(r.date, int(r.hour)): r.app for r in usage.itertuples()}
    X, y = [], []
    for i in range(days):
        dt = end - timedelta(days=days - i)
        for h in range(24):
            if is_asleep(h, bed, wake):
                continue
            X.append(feat(dt.weekday(), h, ss, se, bed, wake))
            y.append(used.get((dt.isoformat(), h), "none"))
    return np.array(X), np.array(y)


def pick_threshold(y, p, target):
    """Highest alert level that still catches at least `target` of the really-used hours."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    if len(ok) == 0:
        return float(thr[0]) if len(thr) else 0.5
    return float(thr[ok[np.argmax(prec[:-1][ok])]])


def effective_threshold(t_star, pref):
    """The saved lighter / same / heavier choice: lighter = only 80%+ sure, heavier = 50%+ sure,
    same = the standard 60%."""
    return 0.8 if pref >= 0.75 else 0.5 if pref <= 0.55 else float(t_star)


def train_usage_model(usage, ss, se, bed, wake, end_iso, days=42):
    """Two models.
      A) 'Will any app be used this hour?'  A random forest whose settings are chosen by cross-validation
         (scored on how good its probabilities are), then CALIBRATED so that '60% sure' really means about
         6 times out of 10. A notification fires at CONFIDENCE (60%). Accuracy is measured with
         cross-validated predictions, so every hour is scored by a model that never trained on it.
      B) 'Which app?'  Trained only on the hours when an app was used."""
    X, y = build_table(usage, ss, se, bed, wake, end_iso, days)
    yb = (y != "none").astype(int)
    Xtr, Xte, ytr, yte, yapp_tr, yapp_te = train_test_split(X, yb, y, test_size=0.25, stratify=yb,
                                                            random_state=1)
    # the old single multi-class forest, kept only for comparison
    old = RandomForestClassifier(n_estimators=100, min_samples_leaf=4, random_state=1).fit(Xtr, yapp_tr)
    baseline_recall = float(np.mean(old.predict(Xte)[yte == 1] != "none"))

    grid = {"min_samples_leaf": [2, 4], "max_features": ["sqrt", None]}
    search = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=1), grid, cv=3,
                          scoring="neg_brier_score")
    search.fit(X, yb)
    best = search.best_estimator_
    calibrated = CalibratedClassifierCV(clone(best), method="sigmoid", cv=5)
    oof = cross_val_predict(calibrated, X, yb, cv=5, method="predict_proba")[:, 1]
    flagged = oof >= CONFIDENCE
    accuracy = float((flagged == (yb == 1)).mean())
    precision = float(yb[flagged].mean()) if flagged.any() else 0.0
    recall = float(flagged[yb == 1].mean())

    final = clone(calibrated).fit(X, yb)                 # refit on everything for the forecast
    plain = clone(best).fit(X, yb)                       # only used to read feature importances
    which = RandomForestClassifier(n_estimators=100, random_state=1).fit(X[yb == 1], y[yb == 1])
    slots = [(d, h) for d in range(7) for h in range(24) if not is_asleep(h, bed, wake)]
    F = np.array([feat(d, h, ss, se, bed, wake) for d, h in slots])
    p_any = final.predict_proba(F)[:, 1]
    app = which.predict(F)
    forecast = {slot: (float(p), str(a)) for slot, p, a in zip(slots, p_any, app)}
    return {"accuracy": accuracy, "recall": recall, "precision": precision, "t_star": CONFIDENCE,
            "ready": accuracy >= MIN_ACCURACY, "baseline_recall": baseline_recall, "forecast": forecast,
            "importances": {n: float(v) for n, v in zip(FEATURE_NAMES, plain.feature_importances_)}}
