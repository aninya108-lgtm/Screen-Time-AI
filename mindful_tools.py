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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

APPS = ["Instagram", "TikTok", "Snapchat", "YouTube"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LOCS = ["home", "school", "outside"]
CONFIDENCE = 0.6      # send a habit notification when the model is at least this sure
MIN_ACCURACY = 0.8    # accuracy goal that is reported next to the measured accuracy
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


def _near(h, c, width):
    """A smooth bump: 1 at hour c, fading with distance (the clock wraps around midnight)."""
    d = abs(h - c) % 24
    d = min(d, 24 - d)
    return math.exp(-0.5 * (d / width) ** 2)


def simulate_usage(ss, se, bed, wake, seed, days=42, end_iso=None):
    """Screen time for a made-up teenager, built to be messy the way real life is.
    Every seed is a different person: how heavy a user, how regular, night owl or not, whether the school
    bans phones, which days they have practice, which apps they favor. On top of that each DAY differs
    (some days are heavy, some light, an occasional sick or busy day), habit times wobble by day, use drifts
    up or down over the weeks, and sessions come in short checks and long binges.
    Returns (usage DataFrame, list of the habits that really showed up, for the answer key)."""
    rng = random.Random(seed)
    end = date.fromisoformat(end_iso) if end_iso else date.today()

    # ---- who this teenager is ----
    heavy = min(1.5, max(0.8, rng.lognormvariate(0.1, 0.25)))     # light to heavy user
    routine = rng.uniform(0.85, 1.0)                              # 1 = very regular, 0.4 = fairly random
    owl = rng.random() < 0.35                                    # stays up scrolling
    phone_ban = rng.random() < 0.5                               # school keeps phones away
    shift = rng.choice([-1.0, -0.5, 0.0, 0.0, 0.5, 1.0])         # their own habit timing
    busy = rng.sample(range(5), rng.randint(0, 2))               # practice / club days
    base0 = rng.uniform(0.02, 0.05)                              # random pickups at any hour
    drift = rng.uniform(-0.2, 0.2)                             # use creeping up or down
    fav = {a: rng.uniform(0.4, 1.6) for a in APPS}
    awake = [h for h in range(24) if not is_asleep(h, bed, wake)]

    def profile(weekend, jit, busy_day):
        """Chance of using an app in each awake hour, for one kind of day."""
        out = {}
        if weekend:
            bumps = [(0.55, 17 + jit[0], 4.0),                    # most of the day is phone time
                     (0.40, (wake + 3 + (1.5 if owl else 0)) % 24 + jit[0], 1.3),
                     (0.55, 15 + jit[1], 2.0),
                     (0.90, 20 + shift + jit[2], 1.5),
                     (0.92 * (1.1 if owl else 1), (bed - 1) % 24 + jit[3], 0.9)]
        else:
            after = 0.95 * (0.25 if busy_day else 1)
            plateau = ((se + bed) / 2 + shift, max(2.0, (bed - se) / 2.2))     # home time: after school until bed
            bumps = [(0.55 if not busy_day else 0.3, plateau[0] + jit[1], plateau[1]),
                     (0.25, (ss - 1) % 24 + jit[0], 0.8),
                     (0.35, (ss + se) / 2 + jit[1], 0.8),
                     (after, se + 0.3 + shift + jit[1], 1.0),
                     (0.65, (bed - 3) % 24 + shift + jit[2], 1.2),
                     (0.92 * (1.1 if owl else 1), (bed - 1) % 24 + jit[3], 0.9)]
            if busy_day:                                         # home from practice, tired, scrolls later
                bumps.append((0.55, se + 2.7 + jit[1], 1.0))
        for h in awake:
            keep = 1 - base0
            for amp, c, w in bumps:
                keep *= 1 - amp * _near(h, c, w)
            p = 1 - keep
            if not weekend and ss <= h < se:                    # in class
                p *= 0.12 if phone_ban else 0.4
            out[h] = p
        mean_p = sum(out.values()) / len(out)
        return {h: routine * p + (1 - routine) * mean_p for h, p in out.items()}

    def pick_app(h, weekend):
        w = dict(fav)
        if ss <= h < se and not weekend:
            w["Snapchat"] *= 1.6
        if h >= bed - 3 or h < wake + 1:
            w["YouTube"] *= 1.6
            w["TikTok"] *= 1.3
        elif h in (se, se + 1):
            w["TikTok"] *= 1.5
        elif h < ss:
            w["Instagram"] *= 1.4
        names = list(w)
        return rng.choices(names, weights=[w[a] for a in names])[0]

    rows = []
    for i in range(days):
        dt = end - timedelta(days=days - i)
        d = dt.weekday()
        weekend = d >= 5
        jit = [rng.gauss(0, 0.3) for _ in range(4)]             # habit times wobble from day to day
        prof = profile(weekend, jit, d in busy)
        mood = rng.lognormvariate(0, 0.12) * (1 + drift * (i / max(1, days - 1) - 0.5))
        odd = rng.random()
        if odd < 0.03:                                           # sick day or away: much less
            mood *= 0.4
        elif odd < 0.06:                                         # sick in bed at home: much more
            mood *= 1.5
        quiet = set()
        if weekend and rng.random() < 0.35:                       # out with friends for a few hours
            start = rng.randint(11, 18)
            quiet = {start, start + 1, start + 2}
        for h in awake:
            p = min(0.95, prof[h] * heavy * mood * (0.3 if h in quiet else 1))
            if rng.random() < p:
                long_binge = rng.random() < 0.15
                mins = min(90, max(2, int(rng.lognormvariate(math.log(13), 0.7) * (1.8 if long_binge else 1))))
                rows.append((dt.isoformat(), d, h, pick_app(h, weekend), mins))
    usage = pd.DataFrame(rows, columns=["date", "weekday", "hour", "app", "minutes"])

    # ---- the answer key: habits that actually showed up in this person's data ----
    shown = []
    for weekend in (False, True):
        n_days = max(1, sum(1 for i in range(days) if ((end - timedelta(days=days - i)).weekday() >= 5) == weekend))
        sub = usage[(usage["weekday"] >= 5) == weekend]
        stats = []
        for h in awake:
            hits = sub[sub["hour"] == h]
            stats.append((hits["date"].nunique() / n_days, h, hits["app"].mode().iat[0] if len(hits) else ""))
        stats.sort(reverse=True)
        strong = [s for s in stats if s[0] >= 0.45][:5] or stats[:2]
        for rate, h, app in sorted(strong, key=lambda s: s[1]):
            shown.append({"slot": "habit", "hour": h, "app": app, "chance": round(rate, 2),
                          "when": "on weekends" if weekend else "on weekdays"})
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


def _awake_count(bed, wake):
    return sum(1 for h in range(24) if not is_asleep(h, bed, wake))


def train_usage_model(usage, ss, se, bed, wake, end_iso, days=42):
    """Two models, tested the way a real forecast is used: on days that come AFTER the ones it learned from.
      A) 'Will any app be used this hour?'  A random forest (settings chosen with time-ordered cross-validation),
         then CALIBRATED so '60% sure' means about 6 times out of 10. A notification fires at CONFIDENCE (60%).
         Accuracy comes from walking forward week by week: learn from the past, predict the next week,
         never peek. It is compared with two simple baselines (always 'no use', and 'same as last week').
      B) 'Which app?'  Trained only on the hours when an app was used."""
    X, y = build_table(usage, ss, se, bed, wake, end_iso, days)
    yb = (y != "none").astype(int)
    per_day = _awake_count(bed, wake)
    day = np.repeat(np.arange(days), per_day)

    grid = {"min_samples_leaf": [3, 6, 10], "max_features": ["sqrt", None]}
    search = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=1), grid,
                          cv=TimeSeriesSplit(n_splits=3), scoring="neg_brier_score")
    search.fit(X, yb)
    best = search.best_estimator_

    # walk forward: learn from days before k, predict days k .. k+6
    start = max(14, days // 3)
    folds = list(range(start, days, 7))
    probs, truth, last_week = [], [], []
    for k in folds:
        tr, te = day < k, (day >= k) & (day < k + 7)
        if te.sum() == 0 or len(set(yb[tr])) < 2:
            continue
        cal = CalibratedClassifierCV(clone(best), method="sigmoid", cv=3).fit(X[tr], yb[tr])
        probs.append(cal.predict_proba(X[te])[:, 1])
        truth.append(yb[te])
        idx = np.where(te)[0]
        last_week.append(yb[idx - 7 * per_day])               # the same hour, one week earlier
    p, t, lw = np.concatenate(probs), np.concatenate(truth), np.concatenate(last_week)
    flagged = p >= CONFIDENCE
    accuracy = float(((p >= 0.5) == (t == 1)).mean())          # standard accuracy: 50/50 cut-off
    accuracy_at_conf = float((flagged == (t == 1)).mean())
    precision = float(t[flagged].mean()) if flagged.any() else 0.0
    recall = float(flagged[t == 1].mean()) if (t == 1).any() else 0.0
    baseline_accuracy = float((t == 0).mean())                 # always say "no use"
    naive_accuracy = float((lw == t).mean())                   # "same as last week"
    naive_recall = float(lw[t == 1].mean()) if (t == 1).any() else 0.0

    final = CalibratedClassifierCV(clone(best), method="sigmoid", cv=5).fit(X, yb)
    plain = clone(best).fit(X, yb)                             # only used to read feature importances
    which = RandomForestClassifier(n_estimators=100, random_state=1).fit(X[yb == 1], y[yb == 1])
    slots = [(d, h) for d in range(7) for h in range(24) if not is_asleep(h, bed, wake)]
    F = np.array([feat(d, h, ss, se, bed, wake) for d, h in slots])
    p_any = final.predict_proba(F)[:, 1]
    app = which.predict(F)
    forecast = {slot: (float(q), str(a)) for slot, q, a in zip(slots, p_any, app)}
    # Nudges are on when the model has hours it is confident about AND those confident calls are right at
    # least as often as it claims (60% sure -> right 60%+ of the time). MIN_ACCURACY is the goal we report.
    ready = bool(precision >= CONFIDENCE and flagged.any())
    return {"accuracy": accuracy, "accuracy_at_conf": accuracy_at_conf, "recall": recall, "precision": precision, "t_star": CONFIDENCE,
            "ready": ready, "meets_goal": bool(accuracy >= MIN_ACCURACY), "baseline_accuracy": baseline_accuracy, "naive_accuracy": naive_accuracy,
            "baseline_recall": naive_recall, "n_test_hours": int(len(t)), "forecast": forecast,
            "importances": {n: float(v) for n, v in zip(FEATURE_NAMES, plain.feature_importances_)}}