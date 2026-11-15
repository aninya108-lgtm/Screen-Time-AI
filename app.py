"""Mindful Scroll: an all-Python web app with a machine learning model.
"""


import json
import os
import random
import time

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
APPS = {"Instagram": ("📸", "#e1306c"), "TikTok": ("🎵", "#00a3a8"),
        "Snapchat": ("👻", "#c9a400"), "YouTube": ("▶️", "#e0002b")}
LOCS = ["home", "school", "outside"]
FILE = "mindful_data.json"
THRESHOLD = 0.5          # warn when the model thinks use is at least 50% likely

HOBBY_IDEAS = [("soccer", "Kick a ball around outside for 10 minutes."),
               ("basketball", "Shoot some hoops."),
               ("drawing", "Sketch something near you for 10 minutes."),
               ("guitar", "Pick up your guitar and play one song."),
               ("music", "Put on an album and just listen."),
               ("reading", "Read a few pages of your book."),
               ("cooking", "Make yourself a snack from scratch."),
               ("gaming", "Play one short level of a game you love."),
               ("danc", "Put on a song and dance it out."),
               ("running", "Go for a quick jog around the block."),
               ("writing", "Write for 10 minutes about anything on your mind.")]
GOAL_IDEAS = [("fit", "Do 20 jumping jacks and 10 push-ups."),
              ("study", "Review your notes for 10 minutes."),
              ("grade", "Do 10 minutes of homework."),
              ("sleep", "Stretch and drink some water to wind down."),
              ("friend", "Text a friend to make plans."),
              ("learn", "Spend 10 minutes learning something new.")]
GENERIC_IDEAS = ["Take a short walk outside.", "Drink a glass of water and stretch.",
                 "Tidy your space for 5 minutes.", "Write down three things you're grateful for."]

CSS = """<style>
.stApp{background:linear-gradient(135deg,#2b1055,#5b2a9e 55%,#c2418a)}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp li{color:#fff !important}
.stButton>button{border:0;border-radius:999px;font-weight:700;color:#fff;background:linear-gradient(90deg,#7c4dff,#ff5fa2)}
.phone{background:linear-gradient(160deg,#2b2160,#0f1b3d);border:6px solid #111;border-radius:38px;padding:18px;max-width:280px}
.clk{text-align:center;color:#fff;font-weight:700;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.ic{position:relative;border-radius:18px;padding:14px 4px;text-align:center;color:#fff;font-size:.85rem}
.ic b{display:block;font-size:2rem}
.ic.lock{filter:grayscale(1) brightness(.55)}
.ic.lock::after{content:"🔒";position:absolute;top:6px;right:8px;font-size:1.2rem}
.ball{width:90px;height:90px;border-radius:50%;margin:20px auto;background:radial-gradient(circle,#ff5fa2,#7c4dff);animation:b 10s ease-in-out infinite}
@keyframes b{0%{transform:scale(1)}40%{transform:scale(1.8)}60%{transform:scale(1.8)}100%{transform:scale(1)}}
</style>"""


# ---------- saving and loading ----------
def load():
    if os.path.exists(FILE):
        with open(FILE) as f:
            return json.load(f)
    return {"profile": None, "hist": [], "declines": 0}


def save(data):
    with open(FILE, "w") as f:
        json.dump(data, f)


# ---------- the machine learning part ----------
def is_asleep(h, bed, wake):
    if bed > wake:
        return h >= bed or h < wake
    return bed <= h < wake


def where(d, h, ss, se):
    """Fake location sensor, based on the school schedule."""
    if d >= 5:
        return "outside" if 13 <= h < 18 else "home"
    return "school" if ss <= h < se else "home"


def make_data(ss, se, bed, wake):
    """One month of simulated tracking: one row per awake hour.
    Features: [day, hour, where the user was an hour earlier]. Label: app used, or 'none'."""
    lunch = (ss + se) // 2
    X, y = [], []
    for day in range(28):
        d = day % 7
        if d >= 5:
            rules = [(11, "YouTube", 60), (15, "Instagram", 50), (20, "TikTok", 75)]
        else:
            rules = [(lunch, "Instagram", 50), (se, "TikTok", 80),
                     (se + 1, "Snapchat", 60), (bed - 1, "YouTube", 70)]
        for h in range(24):
            if is_asleep(h, bed, wake):
                continue
            label = "none"
            for rule_hour, app, percent in rules:
                if rule_hour % 24 == h and random.randint(1, 100) <= percent:
                    label = app
            if label == "none" and random.randint(1, 100) <= 6:    # random noise
                label = random.choice(list(APPS))
            X.append([d, h, LOCS.index(where(d, (h - 1) % 24, ss, se))])
            y.append(label)
    return X, y


@st.cache_resource
def train_model(ss, se, bed, wake):
    """Train a Random Forest, then predict every (day, hour) of the week."""
    X, y = make_data(ss, se, bed, wake)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    model = RandomForestClassifier(n_estimators=100, random_state=1).fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    slots = [(d, h) for d in range(7) for h in range(24) if not is_asleep(h, bed, wake)]
    features = [[d, h, LOCS.index(where(d, (h - 1) % 24, ss, se))] for d, h in slots]
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    none_i = classes.index("none")

    forecast = {}      # (day, hour) -> (chance of using any app, most likely app)
    for slot, probs in zip(slots, probabilities):
        best = max((i for i in range(len(classes)) if i != none_i), key=lambda i: probs[i])
        forecast[slot] = (1 - probs[none_i], classes[best])
    return accuracy, forecast


def get_model(profile):
    return train_model(profile["ss"], profile["se"], profile["bed"], profile["wake"])


def next_moment(forecast, d, h):
    """Find the next (day, hour) where the model expects social media use."""
    for _ in range(7 * 24):
        h += 1
        if h > 23:
            h, d = 0, (d + 1) % 7
        if (d, h) in forecast and forecast[(d, h)][0] >= THRESHOLD:
            return d, h
    return None


def pick_activity(profile):
    ideas = []
    for hobby in profile["hobbies"]:
        ideas += [a for k, a in HOBBY_IDEAS if k in hobby]
    for goal in profile["goals"]:
        ideas += [a for k, a in GOAL_IDEAS if k in goal]
    return random.choice(ideas or GENERIC_IDEAS)


# ---------- app state and callbacks ----------
st.set_page_config(page_title="Mindful Scroll", page_icon="🌈")
st.markdown(CSS, unsafe_allow_html=True)
S = st.session_state
if "data" not in S:
    S.data = load()
    S.stage, S.locked, S.clock, S.msg = None, None, {"d": 0, "h": 8}, None
data = S.data
profile = data["profile"]


def set_stage(stage):
    S.stage = stage


def jump():
    acc, forecast = get_model(S.data["profile"])
    target = next_moment(forecast, S.clock["d"], S.clock["h"])
    if target is None:
        S.msg = "No predicted moments found."
        return
    d, h = target
    S.clock = {"d": (d - 1) % 7 if h == 0 else d, "h": (h - 1) % 24}
    S.target_app = forecast[target][1]
    S.activity = pick_activity(S.data["profile"])
    S.stage = "intro"


def decline_breathing():
    S.data["declines"] += 1
    save(S.data)
    S.stage = "activity"


def finish(result):
    S.data["hist"].append(result)
    save(S.data)
    S.locked, S.stage = None, None
    S.msg = ("Great choice! Your future self says thanks. 💜" if result == "resisted"
             else "Unlocked. Enjoy, and stay mindful of your time.")


def reset_all():
    if os.path.exists(FILE):
        os.remove(FILE)
    st.cache_resource.clear()
    S.clear()


def type_out(text, delay=0.055):
    """Show text one letter at a time."""
    box, shown = st.empty(), ""
    for ch in text:
        shown += ch
        box.markdown("### " + shown)
        time.sleep(delay)


def phone_html():
    minute = "55" if S.stage else "00"
    icons = ""
    for name, (emoji, color) in APPS.items():
        lock = " lock" if S.locked == name else ""
        icons += f'<div class="ic{lock}" style="background:{color}"><b>{emoji}</b>{name}</div>'
    return (f'<div class="phone"><div class="clk">{DAYS[S.clock["d"]]} '
            f'{S.clock["h"]:02d}:{minute}</div><div class="grid">{icons}</div></div>')


@st.fragment(run_every=1)
def countdown():
    left = S.block_end - time.time()
    if left <= 0:
        S.stage = "choice"
        st.rerun()
        return
    st.progress(min(1.0, left / S.block_total))
    st.write(f"Paused for {int(left) + 1} more seconds (demo speed)")


# ---------- survey screens ----------
def survey_screen(full):
    st.title("Mindful Scroll 🌈")
    st.subheader("Let's get to know you" if full else "Update your goals and hobbies")
    old = profile or {"goals": [], "hobbies": [], "ss": 8, "se": 15, "bed": 23, "wake": 7}
    with st.form("survey"):
        goals = st.text_input("Your goals (separate with commas)", ", ".join(old["goals"]))
        hobbies = st.text_input("Your hobbies (separate with commas)", ", ".join(old["hobbies"]))
        if full:
            st.caption("Use 24-hour time (3 pm = 15).")
            c = st.columns(4)
            ss = c[0].number_input("School starts", 0, 23, old["ss"])
            se = c[1].number_input("School ends", 0, 23, old["se"])
            bed = c[2].number_input("Bedtime", 0, 23, old["bed"])
            wake = c[3].number_input("Wake up", 0, 23, old["wake"])
        submitted = st.form_submit_button("Save")
    if submitted:
        g = [x.strip().lower() for x in goals.split(",") if x.strip()]
        b = [x.strip().lower() for x in hobbies.split(",") if x.strip()]
        if not g or not b:
            st.error("Please add at least one goal and one hobby.")
            return
        new = dict(old, goals=g, hobbies=b, at=time.time())
        if full:
            new.update(ss=int(ss), se=int(se), bed=int(bed), wake=int(wake))
        data["profile"] = new
        save(data)
        S.edit = S.monthly = False
        st.rerun()


def keep_going():
    data["profile"]["at"] = time.time()
    save(data)
    st.cache_resource.clear()      # retrain on the new month
    S.monthly = False


if profile is None:
    survey_screen(True)
    st.stop()
if S.get("edit"):
    survey_screen(False)
    st.stop()
if S.get("monthly") or time.time() - profile["at"] > 30 * 86400:
    st.title("Monthly check-in 🗓️")
    st.write(f"Your goals: **{', '.join(profile['goals'])}**. "
             f"Your hobbies: **{', '.join(profile['hobbies'])}**.")
    c1, c2 = st.columns(2)
    c1.button("Keep going as is", on_click=keep_going)
    c2.button("Change goals/hobbies", on_click=lambda: S.update(edit=True))
    st.stop()

# ---------- main screen ----------
accuracy, forecast = get_model(profile)
st.title("Mindful Scroll 🌈")
left, right = st.columns([1, 2])
left.markdown(phone_html(), unsafe_allow_html=True)

with right:
    if S.stage is None:
        if S.msg:
            st.success(S.msg)
            S.msg = None
        resisted = data["hist"].count("resisted")
        st.subheader("Your progress 💪")
        st.write(f"Resisted **{resisted}** of **{len(data['hist'])}** check-ins.")
        st.button("⏩ Jump to next predicted moment", on_click=jump)
        c1, c2, c3 = st.columns(3)
        c1.button("Monthly check-in", on_click=lambda: S.update(monthly=True))
        c2.button("Retrain", on_click=st.cache_resource.clear)
        c3.button("Reset", on_click=reset_all)

        st.subheader("What the model learned 🔍")
        st.write(f"Random Forest accuracy on unseen hours: **{accuracy:.0%}** "
                 "(trained on one simulated month).")
        top = sorted(forecast.items(), key=lambda kv: -kv[1][0])[:5]
        for (d, h), (p, app) in top:
            st.write(f"{DAYS[d]} {h}:00 → {p:.0%} chance of use ({app})")
        st.caption("Demo note: tracking data is simulated, and a web page can't lock "
                   "real phone apps, so the lock is shown on the mock phone.")

    elif S.stage == "intro":
        type_out("Hey, quick pause. How are you feeling right now? Let's take a moment for you. 💜")
        if data["declines"] < 2:
            c1, c2 = st.columns(2)
            c1.button("Yes, let's breathe 🌬️", on_click=set_stage, args=("breathe",))
            c2.button("Not now", on_click=decline_breathing)
        else:
            S.stage = "activity"
            st.rerun()

    elif S.stage == "breathe":
        st.markdown('<div class="ball"></div>', unsafe_allow_html=True)
        box = st.empty()
        for i in range(1, 6):
            box.markdown(f"### Breath {i} of 5: breathe in…")
            time.sleep(4)
            box.markdown("### Hold…")
            time.sleep(2)
            box.markdown("### Breathe out…")
            time.sleep(4)
        box.markdown("### Nice work. 🌿")
        time.sleep(1.5)
        S.stage = "activity"
        st.rerun()

    elif S.stage == "activity":
        type_out("Instead of scrolling, how about this: " + S.activity)
        recent = data["hist"][-3:].count("resisted")
        S.block_total = max(2, 5 - recent) * 3       # shorter pauses as habits improve
        S.block_end = time.time() + S.block_total
        S.locked = S.target_app
        time.sleep(1)
        S.stage = "block"
        st.rerun()

    elif S.stage == "block":
        st.markdown(f"### 🔒 Paused. Try this: {S.activity}")
        countdown()
        st.button("I'm good, I'll skip it ✅", on_click=finish, args=("resisted",))

    elif S.stage == "choice":
        st.markdown("### The pause is over. It's your choice. 💜")
        c1, c2 = st.columns(2)
        c1.button("Unlock the app", on_click=finish, args=("opened",))
        c2.button("I'll skip it ✅", on_click=finish, args=("resisted",))