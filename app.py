<<<<<<< HEAD
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
=======
"""Mindful Scroll (mobile-first).
Install:  pip install streamlit scikit-learn pandas textblob anthropic edge-tts
Run:      streamlit run app.py
Phone test: open the "Network URL" the terminal prints, on a phone on the same Wi-Fi.

Optional AI replies: set the ANTHROPIC_API_KEY environment variable. Without it, the app
falls back to a simpler built-in reply. Put config.toml in a ".streamlit" folder next to
this file so the beige and green theme applies to every widget.
"""
import difflib
import html
import json
import os
import random
import re
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import mindful_tools as mt

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
APPS = ["Instagram", "TikTok", "Snapchat", "YouTube"]
LOCS = ["home", "school", "outside"]
USER_DIR = "mindful_users"           # one small file per visitor (see user_id)
APP_URL = os.environ.get("MINDFUL_URL", "")   # put your public link here once deployed
VOICE = os.environ.get("MINDFUL_VOICE", "en-US-JennyNeural")
LEARNING_DAYS = 42       # six weeks of learning before predictions start
BREATH_SECONDS = 50      # five breaths of 10 seconds
HOURS = [f"{h % 12 or 12} {'am' if h < 12 else 'pm'}" for h in range(24)]
DEFAULT_HOUR = {"ss": 8, "se": 15, "bed": 23, "wake": 7}
FEATURE_NAMES = ["Day of week", "Hour of day", "Is weekend", "Where you just were",
                 "Hours since school starts", "Hours since school ends",
                 "Hours until bedtime", "Hours since wake-up"]
LLM_MODEL = os.environ.get("MINDFUL_MODEL", "claude-haiku-5-5")
LLM_ON = bool(os.environ.get("ANTHROPIC_API_KEY"))
KEEP_ROUTINE_NUDGES = False   # True = keep wake-up/bedtime check-ins after the 42 learning days too
mt.init_db(os.path.join(USER_DIR, "mindful.db"))   # local SQLite file

# name -> (activity idea, how to start)
GOALS = {
    "get fit": ("Do 20 jumping jacks and 10 push-ups.", "Clear a little floor space and start with just 5 jumping jacks."),
    "study more": ("Review your notes for 10 minutes.", "Open your notebook to today’s page and read the first heading."),
    "learn something new": ("Spend 10 minutes learning something new.", "Pick one topic you’re curious about and read or watch one short explainer."),
    "improve my grades": ("Do 10 minutes of homework.", "Pick the easiest assignment and write your name and the date at the top."),
    "sleep better": ("Stretch and drink some water to wind down.", "Stand up, reach your arms overhead, and take one slow breath."),
    "make friends": ("Text a friend to make plans.", "Open your messages and send a simple “Hey, want to hang out?”"),
    "be more creative": ("Doodle or write for 10 minutes.", "Grab any paper and make your first line or word."),
    "get organized": ("Tidy your desk or backpack for 5 minutes.", "Start by clearing one small area."),
    "be more active": ("Take a 10-minute walk outside.", "Put on your shoes and walk to the end of the block."),
    "feel calmer": ("Write down three things on your mind.", "Open a notes page and write the first one."),
    "read more": ("Read 10 pages of a book.", "Open your book to your bookmark and read one paragraph."),
    "learn an instrument": ("Practice a chord or scale for 10 minutes.", "Pick up your instrument and play the first thing you know."),
}
GOAL_LABELS = {
    "get fit": "Improve physical fitness",
    "study more": "Dedicate more time to studying",
    "learn something new": "Learn a new skill",
    "improve my grades": "Raise my academic performance",
    "sleep better": "Improve sleep quality",
    "make friends": "Build new friendships",
    "be more creative": "Cultivate creativity",
    "get organized": "Become more organized",
    "be more active": "Increase daily physical activity",
    "feel calmer": "Cultivate a calmer state of mind",
    "read more": "Read more regularly",
    "learn an instrument": "Learn a musical instrument",
}
HOBBIES = {
    "soccer": ("Kick a ball around outside for 10 minutes.", "Grab a ball and do 10 touches against a wall."),
    "basketball": ("Shoot some hoops.", "Take your first 5 free throws."),
    "drawing": ("Sketch something near you for 10 minutes.", "Pick one object and draw its outline."),
    "painting": ("Paint or color something small.", "Get out one color and make your first stroke."),
    "guitar": ("Pick up your guitar and play one song.", "Tune up, then start with the chord you know best."),
    "piano": ("Play one song on the piano or keyboard.", "Sit down and play the first four notes."),
    "music": ("Put on an album and just listen.", "Pick one album, press play, and put your phone across the room."),
    "reading": ("Read a few pages of your book.", "Open to your bookmark and read one paragraph."),
    "cooking": ("Make yourself a snack from scratch.", "Pick one ingredient and take it out."),
    "baking": ("Bake or plan something sweet.", "Look up one simple recipe and read the ingredients."),
    "gaming": ("Play one short level of a game you love.", "Open the game and set a 15-minute timer."),
    "dancing": ("Put on a song and dance it out.", "Pick a song and start with just your shoulders."),
    "running": ("Go for a quick jog around the block.", "Put on your shoes and walk to the door."),
    "swimming": ("Plan your next swim, or stretch your shoulders.", "Do 10 slow arm circles each way."),
    "writing": ("Write for 10 minutes about anything on your mind.", "Open a page and write one sentence, even a messy one."),
    "photography": ("Take 5 photos of things around you.", "Find the best light near a window."),
    "yoga": ("Do a short stretch or yoga flow.", "Roll out a mat or towel and start with child’s pose."),
    "skateboarding": ("Skate for 15 minutes.", "Grab your board and head outside."),
    "coding": ("Work on a small coding project.", "Open your project and fix one tiny thing."),
    "gardening": ("Water or check on a plant.", "Fill a glass of water and find a plant."),
}
HOBBY_LABELS = {
    "soccer": "Soccer", "basketball": "Basketball", "drawing": "Drawing", "painting": "Painting",
    "guitar": "Guitar", "piano": "Piano", "music": "Listening to music", "reading": "Reading",
    "cooking": "Cooking", "baking": "Baking", "gaming": "Video gaming", "dancing": "Dancing",
    "running": "Running", "swimming": "Swimming", "writing": "Creative writing",
    "photography": "Photography", "yoga": "Yoga", "skateboarding": "Skateboarding",
    "coding": "Computer programming", "gardening": "Gardening",
}
HOBBY_BADGES = {"guitar": "Guitar Hero", "soccer": "Goal Getter", "basketball": "Hoops Hero",
                "drawing": "Sketch Star", "piano": "Piano Pro", "dancing": "Dance Machine",
                "running": "Road Runner", "writing": "Wordsmith", "gaming": "Level Up",
                "coding": "Code Wizard"}
GENERIC = [("Take a short walk outside.", "Put on your shoes and step outside."),
           ("Drink a glass of water and stretch.", "Stand up and reach for the ceiling.")]
WIND_DOWN = [("Stretch gently for a few minutes.", "Stand up and slowly roll your shoulders."),
             ("Read a few pages of a book.", "Put your phone across the room and open your book."),
             ("Write down one good thing from today.", "Open a notes page and write one sentence.")]
MORNING = [("Drink a glass of water.", "Head to the kitchen and fill a glass."),
           ("Stretch for two minutes.", "Stand up and reach for the ceiling."),
           ("Write down one goal for today.", "Open a notes page and write one line.")]

GRATITUDE = [
    "What is one small thing that made you smile today?",
    "Who is someone you’re glad to have in your life right now?",
    "What is one thing your body did for you today that you’re thankful for?",
    "What is a simple comfort you enjoyed recently?",
    "What is something you learned lately that you’re grateful for?",
    "Who helped you recently, even in a small way?",
    "What is one place that makes you feel calm?",
    "What is a meal or snack you’re looking forward to?",
    "What is something in your day that went better than expected?",
    "What is a song, show, or book you’re thankful for right now?",
    "What is one thing about yourself you appreciate today?",
    "Who made you laugh recently?",
    "What is something you have today that you once wished for?",
    "What is one moment of quiet you enjoyed this week?",
    "What is something kind someone said to you lately?",
    "What is a small win you can be grateful for today?",
    "What part of nature did you notice today?",
    "What is one thing you’re grateful to have finished?",
    "Who would you like to thank, even silently?",
    "What is one thing you’re looking forward to this week?",
    "What is a skill you have that you’re thankful for?",
    "What is something warm or cozy you appreciate today?",
    "What is one challenge that taught you something?",
    "What is a memory that still makes you smile?",
    "What is something you’re grateful you didn’t have to rush today?",
]

REFLECT = [
    "What would feel good to do for ten minutes with no screen?",
    "What is one thing you can let go of today?",
    "Who could you check in on in person today?",
    "What does your body need right now?",
    "What is one thing you’re proud of this week?",
    "What helps you feel calm when things get busy?",
    "What is something you want to try soon?",
    "When did you last feel really present?",
    "What would make tomorrow a little easier?",
    "What is one tiny step you could take toward a goal today?",
]
QUOTES = [
    "“Nature does not hurry, yet everything is accomplished.” — Lao Tzu",
    "“No act of kindness, no matter how small, is ever wasted.” — Aesop",
    "“We suffer more often in imagination than in reality.” — Seneca",
    "“A journey of a thousand miles begins with a single step.” — Lao Tzu",
    "“Almost everything will work again if you unplug it for a few minutes, including you.” — Anne Lamott",
    "“You can’t stop the waves, but you can learn to surf.” — Jon Kabat-Zinn",
    "Small pauses add up to a calmer day.",
    "You don’t have to answer every notification right now.",
    "Your attention is yours to spend. Spend it on what matters to you.",
    "Rest is productive.",
    "Today only needs your next small step.",
    "Breathe in what you need, breathe out what you don’t.",
    "Looking up from your screen is a tiny act of courage.",
    "You are more than your feed.",
    "Not every moment needs to be captured to be real.",
    "Be gentle with yourself. You’re doing the best you can.",
]
PROMPTS = ([("Today’s gratitude", q) for q in GRATITUDE] + [("Something to think about", q) for q in REFLECT]
           + [("Today’s thought", q) for q in QUOTES])
random.Random(11).shuffle(PROMPTS)       # mixed order, the same on every run

# Floating "Menu" button that always works, even after the sidebar arrows were used to hide it.
SIDEBAR_JS = """<script>
(function(){
  const d = window.parent.document;
  const old = d.getElementById('ms-menu-btn'); if (old) old.remove();
  if (d.getElementById('ms-menu-btn2')) return;
  const b = d.createElement('button');
  b.id = 'ms-menu-btn2'; b.type = 'button'; b.textContent = '☰ Menu';
  b.setAttribute('aria-label', 'Show or hide the menu');
  b.style.cssText = 'position:fixed;top:.6rem;left:.6rem;z-index:1000001;border:0;border-radius:999px;'
    + 'padding:.45rem .9rem;background:#2F4F3A;color:#F6F0E4;font:600 .85rem sans-serif;'
    + 'box-shadow:0 2px 8px rgba(47,79,58,.3);cursor:pointer;display:none';
  function sb(){ return d.querySelector('[data-testid="stSidebar"]'); }
  function isOpen(){                       // by position on screen, not by a version-specific attribute
    const s = sb(); if (!s) return false;
    if (s.getAttribute('aria-expanded') === 'false') return false;
    const r = s.getBoundingClientRect();
    return r.width > 40 && r.right > 40;
  }
  function firstButton(el){ return el.tagName === 'BUTTON' ? el : (el.querySelector('button') || el); }
  function findBtn(kind){
    const direct = kind === 'open'
      ? ['[data-testid="stExpandSidebarButton"]', '[data-testid="collapsedControl"]']
      : ['[data-testid="stSidebarCollapseButton"]'];
    for (const s of direct){ const el = d.querySelector(s); if (el) return firstButton(el); }
    const re = kind === 'open' ? /expand|open|show/i : /collapse|close|hide/i;
    const skip = ['stSidebar','stSidebarContent','stSidebarUserContent','stSidebarHeader','stSidebarNav'];
    for (const el of d.querySelectorAll('[data-testid*="idebar"],[aria-label*="idebar"]')){
      const id = el.getAttribute('data-testid') || '';
      if (skip.indexOf(id) >= 0) continue;
      if (re.test(id + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')))
        return firstButton(el);
    }
    return null;
  }
  function force(on){                      // last resort if no native button can be found
    let st = d.getElementById('ms-force');
    if (on && !st){
      st = d.createElement('style'); st.id = 'ms-force';
      st.textContent = '[data-testid="stSidebar"]{transform:none !important;margin-left:0 !important;'
        + 'visibility:visible !important;display:block !important;min-width:21rem !important;width:21rem !important}';
      d.head.appendChild(st);
    } else if (!on && st){ st.remove(); }
  }
  function update(){
    const there = sb() || d.querySelector('[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"]');
    b.style.display = there ? 'block' : 'none';
    b.textContent = isOpen() ? '✕ Close menu' : '☰ Menu';
  }
  b.onclick = function(){
    if (d.getElementById('ms-force')){ force(false); }
    else if (isOpen()){ const t = findBtn('close'); if (t) t.click(); else force(false); }
    else { const t = findBtn('open'); if (t) t.click(); else force(true); }
    setTimeout(update, 150); setTimeout(update, 600);
  };
  d.body.appendChild(b);
  setInterval(update, 400);
})();
</script>"""

INTRO = {"habit": "Hey, quick pause. How are you feeling right now? Let’s take a moment for you. 💜",
         "morning": "Good morning! How is your day starting? Let’s take a moment before it gets busy. 🌅",
         "bed": "Winding down soon? How are you feeling tonight? 🌙"}
NOTIF = {"habit": "Quick pause? Take a moment for yourself. 💜",
         "morning": "Good morning! Start the day gently. 🌅",
         "bed": "Winding down soon? Let’s ease into the evening. 🌙"}
YES = {"habit": "Yes, let’s take a moment 💜",
       "morning": "Yes, let’s start the day gently 🌅",
       "bed": "Yes, let’s ease into the evening 🌙"}

# ---------- feelings: sentiment, safety, replies ----------
NEG_WORDS = {"sad", "stressed", "anxious", "anxiety", "overwhelmed", "tired", "exhausted", "lonely",
             "angry", "upset", "worried", "nervous", "depressed", "awful", "terrible", "bad",
             "hate", "scared", "alone", "bored", "worthless", "hopeless", "crying", "panic"}
POS_WORDS = {"good", "great", "happy", "excited", "calm", "relaxed", "fine", "okay", "awesome",
             "amazing", "proud", "better", "energized", "peaceful", "grateful", "love"}
CRISIS_PHRASES = ["kill myself", "suicide", "suicidal", "want to die", "end my life", "end it all",
                  "hurt myself", "self harm", "self-harm", "cut myself", "better off without me",
                  "no reason to live", "don't want to be here", "dont want to be here"]
TOPICS = {"test": "your test", "exam": "your exam", "quiz": "your quiz", "homework": "your homework",
          "grade": "your grades", "friend": "your friends", "fight": "that conflict",
          "argument": "that argument", "parent": "things at home", "mom": "things at home",
          "dad": "things at home", "sleep": "sleep", "tired": "feeling worn out", "game": "your game",
          "practice": "practice", "team": "your team", "school": "school", "work": "work",
          "job": "work", "phone": "your phone", "lonely": "feeling lonely", "bored": "feeling bored"}
LLM_SYSTEM = (
    "You are the gentle check-in voice of Mindful Scroll, an app that helps teenagers and young adults "
    "use social media more mindfully. The person just told you how they feel. Reply in 2 to 3 short, warm "
    "sentences. Respond to the specific things they mention (events, people, feelings) instead of giving "
    "generic comfort, and do not repeat their words back as a list. Do not diagnose, give medical advice, "
    "or lecture about screen time. End by inviting them to take a few slow breaths. If they mention "
    "anything about harming themselves or being in danger, gently encourage them to reach out to a "
    "trusted adult or call or text 988 (US). Plain text only, at most one emoji.")


def is_crisis(text):
    t = text.lower().replace("’", "'")
    return any(p in t for p in CRISIS_PHRASES)


def analyze_feeling(text):
    """Returns 'negative', 'neutral' or 'positive'. TextBlob if installed, else a small word list."""
    words = set(re.findall(r"[a-z']+", text.lower()))
    neg, pos = len(words & NEG_WORDS), len(words & POS_WORDS)
    try:
        from textblob import TextBlob
        score = TextBlob(text).sentiment.polarity
    except Exception:
        score = (pos - neg) / max(1, pos + neg)
    if neg and score > -0.15:       # TextBlob misses a lot of teen stress words
        score = -0.2
    return "negative" if score <= -0.15 else "positive" if score >= 0.25 else "neutral"


def llm_reply(text):
    """A short reply from Claude that speaks to what the person wrote. None if unavailable."""
    if not LLM_ON:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(timeout=8.0, max_retries=0)
        msg = client.messages.create(model=LLM_MODEL, max_tokens=200, system=LLM_SYSTEM,
                                     messages=[{"role": "user", "content": text}])
        out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        return out or None
    except Exception:
        return None


def local_reply(text, mood):
    """Simple fallback when no AI service is available."""
    t = text.lower()
    topic = next((v for k, v in TOPICS.items() if k in t), None)
    if mood == "negative":
        lead = f"It sounds like {topic} is weighing on you. " if topic else "That sounds like a lot to carry. "
        return lead + "Thank you for putting it into words. Let’s slow down together for a few breaths."
    if mood == "positive":
        lead = f"I’m glad to hear it, especially with {topic}. " if topic else "I’m glad to hear that. "
        return lead + "Let’s keep that calm going with a few slow breaths."
    lead = f"Thanks for sharing about {topic}. " if topic else "Thanks for sharing. "
    return lead + "A short pause can help you reset."


# ---------- style ----------
CSS = """<style>
:root{--beige:#F6F0E4;--paper:#FBF8F1;--sage:#DCE7D3;--green:#2F4F3A;--pink:#F5D9DC;--rose:#E8A0AC}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none !important}
[data-testid="stHeader"]{background:transparent !important}
[data-testid="stHeader"] *{color:var(--green) !important}
.stApp{background:var(--beige) !important}
[data-testid="stSidebar"]{background:var(--sage) !important}
[data-testid="stSidebar"] *{color:var(--green)}
.block-container{max-width:430px;padding:1.2rem 1.2rem 3rem}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp li,.stApp label,.stApp summary,.stApp [data-testid="stCaptionContainer"] *{color:var(--green) !important}
.stApp h1,.stApp h2,.stApp h3{font-family:Georgia,'Times New Roman',serif;font-weight:500}
[data-testid="stMetric"] *{color:var(--green) !important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="base-input"],textarea{background:var(--paper) !important;color:var(--green) !important;border-color:rgba(47,79,58,.25) !important}
input,textarea{color:var(--green) !important}
[data-baseweb="tag"]{background:var(--pink) !important}
[data-baseweb="tag"] *{color:var(--green) !important}
[data-baseweb="popover"] *{color:var(--green) !important}
[data-baseweb="popover"] ul,[data-baseweb="menu"]{background:var(--paper) !important}
.stApp .stButton>button{border:0;border-radius:999px;padding:.6rem 1.4rem;font-weight:600;background:var(--green) !important;transition:background .2s}
.stApp .stButton>button:hover{background:#3f6a4e !important}
.stApp .stButton button p{color:var(--beige) !important}
.stApp .stButton>button:disabled{opacity:.4}
[data-testid="stSidebar"] .stButton>button{width:100%}
[class*="st-key-big_"] .stButton,[class*="st-key-notif_"] .stButton,.st-key-focus_now .stButton,.st-key-quiet_end .stButton{display:flex;justify-content:center}
[class*="st-key-big_"] button,[class*="st-key-notif_"] button,.st-key-focus_now button{width:min(100%,320px);min-height:3.2rem;font-size:1.05rem}
.st-key-notif_yes button,.st-key-notif_no button{width:min(100%,360px);min-height:3.8rem;font-size:1.15rem}
.stApp .st-key-big_no button,.stApp .st-key-notif_no button{background:var(--pink) !important}
.stApp .st-key-big_no button p,.stApp .st-key-notif_no button p{color:var(--green) !important}
.stApp .st-key-quiet_end button{background:transparent !important;border:1px solid rgba(47,79,58,.3);font-weight:400;min-height:2rem;padding:.2rem 1rem}
.stApp .st-key-quiet_end button p{color:var(--green) !important;font-size:.85rem}
.brand{font-size:.72rem;letter-spacing:.3em;text-transform:uppercase;opacity:.5;text-align:center;margin:-.6rem 0 1rem;color:var(--green)}
.day-card{background:var(--sage);border-radius:26px;padding:16px 18px 14px;text-align:center;border:1px solid rgba(47,79,58,.12);color:var(--green);margin-bottom:2.2rem}
.day-card .leaf{font-size:1.3rem}
.day-card .big{font-family:Georgia,serif;font-size:1.45rem;letter-spacing:.02em}
.day-card .sub{font-size:.78rem;letter-spacing:.2em;text-transform:uppercase;opacity:.7;margin-top:2px}
.dots{margin:10px 0 8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:rgba(47,79,58,.18);margin:0 5px}
.dot.done{background:var(--green)}
.dot.now{background:var(--rose);box-shadow:0 0 0 4px rgba(232,160,172,.35)}
.bar{height:4px;border-radius:4px;background:rgba(47,79,58,.15);overflow:hidden}
.bar div{height:100%;background:var(--green);border-radius:4px}
.label{text-align:center;font-size:.7rem;letter-spacing:.25em;text-transform:uppercase;opacity:.6;color:var(--green)}
.prompt{text-align:center;font-family:Georgia,serif;font-style:italic;font-size:1.4rem;line-height:1.5;margin:.6rem 0 2.4rem;color:var(--green)}
.minutes{text-align:center;margin:0 0 2.4rem;color:var(--green)}
.minutes .n{font-family:Georgia,serif;font-size:3.6rem;line-height:1.1}
.minutes .l{opacity:.7;font-size:.85rem}
.say{font-family:Georgia,serif;font-size:1.5rem;line-height:1.4;color:var(--green);margin:.4rem 0 1rem;animation:fade 1.2s}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.msg{background:var(--pink);border-radius:16px;padding:12px 16px;color:var(--green);margin-bottom:14px;animation:fade .8s}
.judge{background:var(--sage);border:1px dashed rgba(47,79,58,.4);border-radius:14px;padding:10px 14px;font-size:.85rem;color:var(--green);margin:10px 0}
.chip{display:inline-block;border-radius:999px;padding:4px 12px;margin:3px;background:var(--paper);color:var(--green);font-size:.85rem}
.chip.lock{background:var(--pink)}
.tgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}
.trophy{background:var(--paper);border:1px solid rgba(47,79,58,.12);border-radius:18px;padding:12px 8px;text-align:center;color:var(--green)}
.trophy.locked{opacity:.45;filter:grayscale(1)}
.trophy.new{background:var(--pink);border-color:var(--rose)}
.trophy .ic{font-size:2rem}
.notif{background:var(--paper);border-radius:22px;padding:16px 20px;margin:14vh auto 1.8rem;width:min(100%,380px);box-shadow:0 10px 30px rgba(47,79,58,.18);animation:fade .6s;color:var(--green)}
.notif,.notif *{color:var(--green) !important}
[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"]{background:var(--sage) !important;border-radius:12px;padding:.25rem .5rem}
.confetti{position:relative;height:0;z-index:5;pointer-events:none}
.confetti i{position:absolute;top:0;left:50%;width:7px;height:11px;border-radius:2px;opacity:0;animation:pop 1.7s ease-out forwards}
@keyframes pop{0%{transform:translate(0,0) rotate(0);opacity:1}100%{transform:translate(var(--x),var(--y)) rotate(var(--r));opacity:0}}
.badge{position:relative;text-align:center;padding:4px 0 2px;animation:reveal 1.1s cubic-bezier(.2,1.4,.4,1)}
@keyframes reveal{0%{transform:scale(.2) rotate(-25deg);opacity:0}100%{transform:none;opacity:1}}
.medal{position:relative;z-index:2;width:116px;height:116px;border-radius:50%;margin:0 auto;padding:8px;box-sizing:border-box;background:conic-gradient(from 200deg,#c9a24b,#f4e2a1,#b88a2e,#f4e2a1,#c9a24b);box-shadow:0 6px 16px rgba(47,79,58,.3)}
.medal .in{width:100%;height:100%;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2.9rem;background:radial-gradient(circle at 35% 30%,#FBF8F1,#DCE7D3);border:2px dashed rgba(47,79,58,.35)}
.ribbon{position:absolute;top:86px;left:0;right:0;display:flex;justify-content:center;gap:12px;z-index:1}
.ribbon i{width:26px;height:58px;background:var(--rose);clip-path:polygon(0 0,100% 0,100% 100%,50% 82%,0 100%)}
.ribbon i:first-child{transform:rotate(12deg)}.ribbon i:last-child{transform:rotate(-12deg)}
.bname{font-family:Georgia,serif;font-size:1.25rem;margin-top:30px;color:var(--green)}
.bdesc{font-size:.8rem;opacity:.7;color:var(--green)}
.medal.sm{width:68px;height:68px;padding:5px}.medal.sm .in{font-size:1.7rem}
.medal.lock{filter:grayscale(1);opacity:.5}
.trophy .medal{margin-bottom:8px}
.st-key-badge_wrap{position:relative}
.st-key-badge_wrap .stButton{position:absolute;inset:0;z-index:5}
.st-key-badge_wrap .stButton button{width:100% !important;height:100% !important;min-height:0 !important;opacity:0 !important}
.block-container{padding-top:4.6rem !important}
[data-testid="stExpandSidebarButton"],[data-testid="collapsedControl"]{display:flex !important}
.st-key-quiet_prompt{margin:-1.6rem 0 1.4rem}
.st-key-quiet_prompt .stButton{display:flex;justify-content:center}
.stApp .st-key-quiet_prompt button{background:transparent !important;border:1px solid rgba(47,79,58,.3);font-weight:400;min-height:2rem;padding:.2rem 1rem}
.stApp .st-key-quiet_prompt button p{color:var(--green) !important;font-size:.85rem}
.focus-note{text-align:center;font-size:.85rem;line-height:1.45;opacity:.8;color:var(--green);margin:-1.4rem auto 1rem;max-width:320px}
.hero{text-align:center;padding:1.2rem 0 .4rem;position:relative;overflow:hidden}
.hero .orb{position:absolute;border-radius:50%;opacity:.6;z-index:0}
.hero .o1{width:150px;height:150px;left:-50px;top:0;background:radial-gradient(circle at 35% 35%,#DCE7D3,#9DBB9A)}
.hero .o2{width:110px;height:110px;right:-35px;top:150px;background:radial-gradient(circle at 35% 35%,#FBE7E9,#E8A0AC)}
.hero .leaf{position:relative;z-index:1;font-size:4.6rem;display:inline-block;animation:floaty 4s ease-in-out infinite}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(-6deg)}50%{transform:translateY(-12px) rotate(6deg)}}
.hero .ttl{position:relative;z-index:1;font-family:Georgia,serif;font-size:2.5rem;line-height:1.1;margin:.3rem 0 .6rem;background:linear-gradient(90deg,#2F4F3A,#5f8f6d 55%,#d98a97);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;color:transparent}
.hero .tag{position:relative;z-index:1;font-family:Georgia,serif;font-style:italic;font-size:1.1rem;color:var(--green);margin-bottom:1.2rem}
.hero .pills{position:relative;z-index:1;margin-bottom:2rem}
.hero .pills span{display:inline-block;margin:3px;padding:6px 14px;border-radius:999px;background:var(--paper);border:1px solid rgba(47,79,58,.15);font-size:.85rem;color:var(--green)}
.clock{position:fixed;top:.55rem;right:.6rem;z-index:1000000;width:178px;background:var(--paper);border:1px solid rgba(47,79,58,.15);border-radius:16px;padding:8px 11px;box-shadow:0 4px 14px rgba(47,79,58,.18);font-size:.72rem;line-height:1.35}
.clock,.clock *{color:var(--green) !important}
.clock .t{font-family:Georgia,serif;font-size:1rem}
.clock .trk{position:relative;height:6px;border-radius:6px;background:rgba(47,79,58,.15);margin:8px 0 4px}
.clock .trk .sch{position:absolute;top:0;height:6px;background:rgba(157,187,154,.8);border-radius:6px}
.clock .trk .mk{position:absolute;top:-3px;width:2px;height:12px;background:var(--green);border-radius:2px;opacity:.55}
.clock .trk .nx{position:absolute;top:-4px;width:4px;height:14px;background:var(--rose);border-radius:2px}
.clock .trk .nw{position:absolute;top:-4px;width:14px;height:14px;margin-left:-7px;border-radius:50%;background:var(--green);border:2px solid var(--paper)}
.clock .nxt{opacity:.9}.clock .jn{margin-top:4px;padding-top:4px;border-top:1px dashed rgba(47,79,58,.3)}
[data-testid="stToolbar"]{display:flex !important;visibility:visible !important}
[data-testid="stToolbarActions"],[data-testid="stMainMenu"],[data-testid="stAppDeployButton"],[data-testid="stStatusWidget"]{display:none !important}
[class*="st-key-big_"],[class*="st-key-notif_"],.st-key-focus_now,.st-key-quiet_prompt,.st-key-quiet_end,.st-key-home_trophy{display:flex !important;justify-content:center !important}
.focus-note{margin:.9rem auto 1.2rem !important}
.stApp .st-key-home_trophy button{background:transparent !important;border:1px solid rgba(47,79,58,.3);font-weight:400;min-height:2rem;padding:.2rem 1rem}
.stApp .st-key-home_trophy button p{color:var(--green) !important;font-size:.85rem}
.st-key-focus_now,.st-key-home_trophy,[class*="st-key-big_"],[class*="st-key-notif_"],.st-key-quiet_prompt,.st-key-quiet_end{width:100% !important;display:flex !important;justify-content:center !important;align-items:center !important;text-align:center}
.st-key-focus_now .stButton,.st-key-home_trophy .stButton{width:auto !important;margin:0 auto !important}
.st-key-focus_now button,.st-key-home_trophy button{min-width:min(100%,260px);margin:0 auto}
</style>"""

SOUND_JS = """
let ctx=null, master=null, nodes=[], timer=null, step=0;
const vol=document.getElementById('vol');
const chords=[[130.81,196,261.63,329.63],[110,164.81,220,261.63],[98,146.83,196,246.94],[98,146.83,196,293.66]];
function level(){ return parseFloat(vol.value)*0.5; }
function init(){
  if(!ctx){
    ctx=new (window.AudioContext||window.webkitAudioContext)();
    master=ctx.createGain(); master.gain.value=0; master.connect(ctx.destination);
  }
  return ctx.resume();
}
function fadeTo(v,s){ if(master) master.gain.setTargetAtTime(v, ctx.currentTime, s||0.8); }
function stopAll(){
  fadeTo(0,0.3);
  const old=nodes; nodes=[];
  setTimeout(()=>old.forEach(n=>{ try{ n.stop(); }catch(e){} }),1500);
  if(timer){ clearInterval(timer); timer=null; }
  document.querySelectorAll('.pill').forEach(b=>b.classList.remove('on'));
}
function noise(kind,hpF,lpF,g){
  const len=ctx.sampleRate*4, buf=ctx.createBuffer(1,len,ctx.sampleRate), d=buf.getChannelData(0);
  let last=0;
  for(let i=0;i<len;i++){
    const w=Math.random()*2-1;
    if(kind==='brown'){ last=(last+0.02*w)/1.02; d[i]=last*3.5; } else { d[i]=w; }
  }
  const ramp=Math.floor(ctx.sampleRate*0.05);
  for(let i=0;i<ramp;i++){ const k=i/ramp; d[i]*=k; d[len-1-i]*=k; }
  const src=ctx.createBufferSource(); src.buffer=buf; src.loop=true;
  const hp=ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value=hpF;
  const lp=ctx.createBiquadFilter(); lp.type='lowpass'; lp.frequency.value=lpF;
  const gn=ctx.createGain(); gn.gain.value=g;
  src.connect(hp); hp.connect(lp); lp.connect(gn); gn.connect(master);
  src.start(); nodes.push(src);
}
function chord(){
  const t=ctx.currentTime, fr=chords[step++ % chords.length];
  fr.forEach(f=>{
    const o=ctx.createOscillator(); o.type='sine'; o.frequency.value=f;
    const lp=ctx.createBiquadFilter(); lp.type='lowpass'; lp.frequency.value=700;
    const g=ctx.createGain();
    g.gain.setValueAtTime(0,t);
    g.gain.linearRampToValueAtTime(0.05,t+1.2);
    g.gain.exponentialRampToValueAtTime(0.001,t+5);
    o.connect(lp); lp.connect(g); g.connect(master); o.start(t); o.stop(t+5.1);
  });
}
function play(name){
  init(); stopAll();
  const b=document.getElementById(name); if(b) b.classList.add('on');
  if(name==='rain') noise('white',500,3600,0.3);
  else if(name==='soft') noise('brown',20,1100,0.9);
  else if(name==='lofi'){ noise('white',3000,6000,0.01); chord(); timer=setInterval(chord,4000); }
  fadeTo(level(),1.2);
}
['rain','soft','lofi'].forEach(n=>{
  document.getElementById(n).onclick=()=>{ play(n); if(window.onEnable) window.onEnable(); };
});
document.getElementById('stop').onclick=stopAll;
vol.addEventListener('input',()=>fadeTo(level(),0.1));
"""

PILLS = """
<div class="row">
  <button class="pill" id="rain">🌧️ Rain</button><button class="pill" id="soft">🌫️ Soft noise</button>
  <button class="pill" id="lofi">🎹 Lo-fi</button><button class="pill" id="stop">Off</button>
</div>
<input id="vol" type="range" min="0" max="1" step="0.05" value="0.4" aria-label="Volume">
"""

SOUND_STYLE = """
<style>
body{margin:0;background:transparent;font-family:Georgia,serif;color:#2F4F3A;text-align:center}
.lbl{font-size:12px;opacity:.7;margin:2px 0 6px;letter-spacing:.1em}
.row{display:flex;gap:6px;flex-wrap:wrap}
.pill{flex:1;border:0;border-radius:999px;padding:8px 6px;font-size:13px;background:#DCE7D3;color:#2F4F3A;cursor:pointer}
.pill.on{background:#2F4F3A;color:#F6F0E4}
input[type=range]{width:100%;margin-top:8px;accent-color:#2F4F3A}
</style>
"""

SOUND_HTML = (SOUND_STYLE + '<div class="lbl">CALMING SOUNDS</div>' + PILLS
              + "<script>" + SOUND_JS + "</script>")

BREATHE_BODY = """
<style>
#count{font-size:13px;letter-spacing:.2em;text-transform:uppercase;opacity:.7;margin-top:6px}
.stage{height:190px;display:flex;align-items:center;justify-content:center}
.ball{width:90px;height:90px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#FBE7E9,#F5D9DC 45%,#9DBB9A);
  box-shadow:0 0 40px rgba(157,187,154,.6);animation:b 10s ease-in-out 5 forwards}
@keyframes b{0%{transform:scale(1)}40%{transform:scale(1.7)}60%{transform:scale(1.7)}100%{transform:scale(1)}}
#cue{font-size:26px;margin:2px 0 8px;min-height:34px}
#lock{display:inline-block;font-size:13px;padding:6px 14px;border-radius:999px;background:#F5D9DC;cursor:pointer;margin-bottom:8px}
</style>
<div id="count">Breath 1 of 5</div>
<div class="stage"><div class="ball"></div></div>
<div id="cue">Breathe in…</div>
<div id="lock">🔈 Tap to start the voice and sound</div>
"""

BREATHE_JS = """
const BREATH=50; let audioOn=false, idx=-1, faded=false; const CLIPS=__CLIPS__;
const cues=[];
for(let i=0;i<5;i++){
  cues.push({t:i*10,text:'Breathe in',show:'Breathe in…',n:i+1});
  cues.push({t:i*10+4,text:'Hold',show:'Hold…',n:i+1});
  cues.push({t:i*10+6,text:'Breathe out',show:'Breathe out…',n:i+1});
}
cues.push({t:50,text:'Nice work',show:'Nice work 🌿',n:5});
function pickVoice(){
  if(!window.speechSynthesis) return null;
  const vs=speechSynthesis.getVoices().filter(v=>v.lang&&v.lang.startsWith('en'));
  const pref=['Samantha','Karen','Moira','Google UK English Female','Microsoft Aria','Microsoft Jenny'];
  for(const p of pref){ const f=vs.find(v=>v.name.includes(p)); if(f) return f; }
  return vs[0]||null;
}
const KIND={'Breathe in':'in','Hold':'hold','Breathe out':'out','Nice work':'done'};
let curAudio=null, curN=1;
function speak(text){
  if(!audioOn) return;
  const k=KIND[text];
  if(CLIPS&&CLIPS[k]){
    const arr=CLIPS[k];
    if(curAudio){ try{curAudio.pause();}catch(e){} }
    curAudio=new Audio('data:audio/mpeg;base64,'+arr[(curN-1)%arr.length]);
    curAudio.volume=0.85; curAudio.play().catch(()=>{});
    return;
  }
  if(!window.speechSynthesis) return;
  const u=new SpeechSynthesisUtterance(text);
  u.rate=0.6; u.pitch=0.85; u.volume=0.6;
  const v=pickVoice(); if(v) u.voice=v;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}
function show(c){
  curN=c.n; document.getElementById('cue').textContent=c.show;
  document.getElementById('count').textContent='Breath '+c.n+' of 5';
}
function enable(){
  if(audioOn) return;
  audioOn=true;
  document.getElementById('lock').style.display='none';
  if(idx>=0) speak(cues[idx].text);
}
window.onEnable=enable;
document.getElementById('lock').onclick=()=>{ play('rain'); enable(); };
try{
  init().then(()=>{ if(ctx.state==='running'){ play('rain'); enable(); } });
}catch(e){}
const t0=performance.now();
const loop=setInterval(()=>{
  const t=(performance.now()-t0)/1000;
  while(idx+1<cues.length && cues[idx+1].t<=t){ idx++; show(cues[idx]); speak(cues[idx].text); }
  if(t>=BREATH-3 && !faded){ faded=true; if(ctx) fadeTo(0,0.8); }
  if(t>BREATH+3){ clearInterval(loop); }
},200);
"""

BREATHE_HTML = (SOUND_STYLE + BREATHE_BODY + PILLS
                + "<script>" + SOUND_JS + BREATHE_JS + "</script>")


VOICE_LINES = {
    "in": ["Breathe in slowly", "Breathe in, nice and deep", "Breathe in, gently"],
    "hold": ["Hold"],
    "out": ["And breathe out", "Let it all go"],
    "done": ["Beautiful. Well done."],
}


@st.cache_resource(show_spinner=False)
def voice_clips():
    """Soft, human-sounding voice clips for the breathing guide. Uses free Microsoft neural voices
    through the edge-tts package (needs internet). Returns None if unavailable, and the page then
    falls back to the browser's built-in voice."""
    try:
        import asyncio
        import base64
        import edge_tts

        async def speak(text):
            audio = b""
            async for chunk in edge_tts.Communicate(text, VOICE, rate="-30%", pitch="-6Hz").stream():
                if chunk["type"] == "audio":
                    audio += chunk["data"]
            return base64.b64encode(audio).decode()

        async def build():
            return {k: [await speak(t) for t in lines] for k, lines in VOICE_LINES.items()}

        return asyncio.run(build())
    except Exception:
        return None


# ---------- saving and loading ----------
def new_data():
    return {"profile": None, "hist": [], "declines": 0, "day0": time.time(),
            "skip_days": 0, "offset_min": 0, "next_monthly": 30, "threshold": 0.65,
            "breaths": 0, "done": {}, "mindful_min": 0.0}


def user_id():
    """Each visitor gets a random id kept in the page address (?u=...). Their data stays theirs,
    and opening the plain address starts fresh."""
    uid = str(st.query_params.get("u", ""))
    if not re.fullmatch(r"[0-9a-f]{16}", uid):
        uid = uuid.uuid4().hex[:16]
        st.query_params["u"] = uid
    return uid


def data_path():
    os.makedirs(USER_DIR, exist_ok=True)
    return os.path.join(USER_DIR, S.uid + ".json")


def load():
    data = new_data()
    if os.path.exists(data_path()):
        with open(data_path()) as f:
            data.update(json.load(f))
    return data


def save(data):
    with open(data_path(), "w") as f:
        json.dump(data, f)


def day_number(data):
    return (time.time() - data["day0"]) / 86400 + data["skip_days"] + data.get("offset_min", 0) / 1440


# ---------- streaks and badges ----------
def current_streak(hist):
    n = 0
    for result in reversed(hist):
        if result != "resisted":
            break
        n += 1
    return n


def best_streak(hist):
    best = run = 0
    for result in hist:
        run = run + 1 if result == "resisted" else 0
        best = max(best, run)
    return best


def badge_list(data):
    """Returns (icon, name, description, have, need) for every badge."""
    hist = data["hist"]
    resisted = hist.count("resisted")
    best = best_streak(hist)
    badges = [
        ("🌬️", "First Breath", "Finish 1 breathing break", data["breaths"], 1),
        ("🧘", "Zen Master", "5 breaks completed", data["breaths"], 5),
        ("🌿", "Calm Collector", "25 breaks completed", data["breaths"], 25),
        ("🛡️", "First Skip", "Skip a scroll once", resisted, 1),
        ("💪", "Strong Will", "Skip 10 scrolls", resisted, 10),
        ("🔥", "On a Roll", "3 skips in a row", best, 3),
        ("🚀", "Unstoppable", "7 skips in a row", best, 7),
    ]
    profile = data["profile"]
    for hobby in (profile["hobbies"] if profile else []):
        name = HOBBY_BADGES.get(hobby, hobby.title() + " Pro")
        word = "practice" if hobby in ("guitar", "piano") else hobby
        badges.append(("🎸" if hobby == "guitar" else "🏅", name, f"3 {word} sessions",
                       data["done"].get(hobby, 0), 3))
    return badges


def earned_names(data):
    return {b[1] for b in badge_list(data) if b[3] >= b[4]}


# ---------- the machine learning part (the code lives in mindful_tools.py) ----------
@st.cache_resource(show_spinner=False)
def usage_data(ss, se, bed, wake, seed, end_iso):
    """Random, schedule-based screen time. A new seed (every visit) gives new habits."""
    return mt.simulate_usage(ss, se, bed, wake, seed, end_iso=end_iso)


def get_usage(profile):
    """Returns (usage table, planted habits) and makes sure it is saved in the SQLite file."""
    p = profile
    usage, rules = usage_data(p["ss"], p["se"], p["bed"], p["wake"], S.sim_seed,
                              datetime.now().date().isoformat())
    key = (S.sim_seed, p["ss"], p["se"], p["bed"], p["wake"])
    if S.get("logged_seed") != key:
        mt.store_usage(S.uid, usage)
        S.logged_seed = key
    return usage, rules


@st.cache_resource(show_spinner="Learning your routine…")
def train_model(ss, se, bed, wake, seed, end_iso):
    usage, _ = usage_data(ss, se, bed, wake, seed, end_iso)
    return mt.train_usage_model(usage, ss, se, bed, wake, end_iso)


def get_model(profile):
    get_usage(profile)
    return train_model(profile["ss"], profile["se"], profile["bed"], profile["wake"], S.sim_seed,
                       datetime.now().date().isoformat())


def eff_threshold(report):
    """Alert level: what the model learned, nudged by the monthly lighter/heavier choice."""
    return mt.effective_threshold(report["t_star"], S.data["threshold"])


def top_patterns(forecast, threshold, limit=3):
    groups = {}
    for (d, h), (p, app) in forecast.items():
        if p >= threshold:
            groups.setdefault((h, app), []).append(d)
    sentences = []
    for (h, app), days in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:limit]:
        days.sort()
        if days == [0, 1, 2, 3, 4]:
            when = "on weekdays"
        elif days == [5, 6]:
            when = "on weekends"
        elif len(days) == 7:
            when = "every day"
        else:
            when = "on " + ", ".join(DAYS[d] for d in days)
        sentences.append(f"**{app}** around **{HOURS[h]}** {when}")
    return sentences


def pick_activity(profile, kind):
    """Returns (idea, how to start, hobby tag or None)."""
    if kind == "bed":
        return (*random.choice(WIND_DOWN), None)
    if kind == "morning":
        return (*random.choice(MORNING), None)
    ideas = [(*GOALS[g], None) for g in profile["goals"] if g in GOALS]
    ideas += [(*HOBBIES[h], h) for h in profile["hobbies"] if h in HOBBIES]
    return random.choice(ideas or [(*g, None) for g in GENERIC])


# ---------- the agent: decides by itself, using the clock ----------
def in_window(n, start_min, length):
    """True when the time of day of n is within `length` minutes after `start_min` (minutes after midnight)."""
    return ((n.hour * 60 + n.minute) - start_min) % 1440 < length


def due_moment(n, data):
    """Should the agent check in at time n? Returns (kind, app) or None.
    First 42 days: a mindful check-in an hour after waking up, and another before bed.
    After that: only when the model is at least 60% sure an app is about to be used, and only
    once its measured accuracy is at least 80%."""
    p = data["profile"]
    learning = day_number(data) < LEARNING_DAYS
    if learning or KEEP_ROUTINE_NUDGES:
        if in_window(n, p["wake"] * 60 + 60, 30):          # an hour after waking up
            return "morning", None
        if in_window(n, p["bed"] * 60 - 30, 30):           # the half hour before bed
            return "bed", None
    if not learning and n.minute >= 55:
        nh = (n.hour + 1) % 24
        nd = (n.weekday() + 1) % 7 if nh == 0 else n.weekday()
        rep = get_model(p)
        slot = rep["forecast"].get((nd, nh))
        if rep["ready"] and slot and slot[0] >= mt.effective_threshold(rep["t_star"], data["threshold"]):
            return "habit", slot[1]
    return None


# ---------- app state and callbacks ----------
st.set_page_config(page_title="Mindful Scroll", page_icon="🌿", layout="centered",
                   initial_sidebar_state="expanded")
S = st.session_state
if "data" not in S:
    S.uid = user_id()
    S.data = load()
    S.started = S.data["profile"] is not None
    S.stage, S.msg, S.offset_min, S.stopped, S.last_key, S.kind = None, None, 0, False, None, None
    S.flow, S.mq, S.steps, S.si, S.draft, S.after, S.target_app = None, None, [], 0, {}, "new", None
    S.page, S.mood, S.reply, S.pending = "main", None, None, None
    S.breathed, S.celebrate, S.new_badges, S.just_earned = False, False, [], []
    S.badges_before, S.badge_until, S.breathe_end = None, 0.0, 0.0
    S.sim_seed, S.logged_seed = random.randrange(2 ** 31), None     # new random screen time on every visit
    S.jump_note, S.jump_until, S.nc_key, S.nc_val = None, 0.0, None, None
    S.prompt_shift, S.agent_text, S.agent_report, S.log_result = 0, None, None, None
data = S.data
profile = data["profile"]
st.markdown(CSS, unsafe_allow_html=True)
components.html(SIDEBAR_JS, height=0)


def now():
    """The agent's clock: real time plus any days/hours the judge has fast-forwarded."""
    return datetime.now() + timedelta(days=S.data["skip_days"], minutes=S.data.get("offset_min", 0))


def fmt_time(t):
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'am' if t.hour < 12 else 'pm'}"


def log(category, number, label=""):
    """Every new piece of data is forced into the strict schema first (mt.extract_record),
    and only then saved to the SQLite file. It never lives only in the chat history."""
    try:
        n = now()
        mt.log_event(S.uid, {"date": n.date().isoformat(), "hour": n.hour, "category": category,
                             "label": label, "number": number})
    except Exception as e:                      # saving must never break the screen
        S.last_log_error = str(e)
>>>>>>> 6c44468 (Mindful Scroll: final hackathon version)


def set_stage(stage):
    S.stage = stage


<<<<<<< HEAD
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
=======
def go_page(page):
    if page == "trophies":
        S.just_earned, S.new_badges = list(S.new_badges), []
    else:
        S.just_earned = []
    S.page = page


def start_intro():
    S.stage, S.mood, S.reply, S.pending = "intro", None, None, None
    S.feel_text = ""
    S.badges_before = earned_names(S.data)


def focus_now():
    """Manual trigger: start a check-in right now."""
    S.kind, S.target_app = "habit", None
    S.activity = pick_activity(S.data["profile"], "habit")
    S.breathed = False
    start_intro()


def share_feeling():
    """Reads the feeling text. It is only kept in memory for this one reply, never saved."""
    text = S.get("feel_text", "").strip()
    S.mood, S.reply, S.pending = None, None, None
    if text and is_crisis(text):
        S.stage = "support"
    elif text:
        S.pending = text[:500]
        S.mood = analyze_feeling(text)
        log("feeling", {"negative": -1, "neutral": 0, "positive": 1}[S.mood], S.mood)   # the words are not saved
        S.stage = "reflecting"
    else:
        S.stage = "offer"


def enter_block():
    recent = S.data["hist"][-3:].count("resisted")
    minutes = max(2, 5 - recent)               # shorter blocks as habits improve
    S.block_total = minutes * 3                # demo speed: 1 minute = 3 seconds
    S.block_end = time.time() + S.block_total
    S.stage = "block"


def go_activity():
    if S.kind == "habit":
        enter_block()
    else:
        S.stage = "activity"


def start_breathing():
    S.stage, S.breathe_end = "breathe", time.time() + BREATH_SECONDS + 2.5


def end_breathing_early():
    go_activity()
>>>>>>> 6c44468 (Mindful Scroll: final hackathon version)


def decline_breathing():
    S.data["declines"] += 1
    save(S.data)
<<<<<<< HEAD
    S.stage = "activity"


def finish(result):
    S.data["hist"].append(result)
    save(S.data)
    S.locked, S.stage = None, None
=======
    go_activity()


def finish_habit(result):
    S.data["hist"].append(result)
    log("scroll_skipped" if result == "resisted" else "scroll_opened", 1, S.target_app or "")
    if result == "resisted":
        tag = S.activity[2]
        if tag:
            S.data["done"][tag] = S.data["done"].get(tag, 0) + 1
        total = S.get("block_total", 0)
        elapsed = min(max(0.0, time.time() - (S.get("block_end", 0) - total)), total)
        S.data["mindful_min"] += elapsed / 3.0
        log("mindful_minutes", elapsed / 3.0, "blocked time")
        if S.breathed:                 # breathing done AND scroll skipped
            S.celebrate = True
    S.breathed = False
    save(S.data)
    S.stage = None
>>>>>>> 6c44468 (Mindful Scroll: final hackathon version)
    S.msg = ("Great choice! Your future self says thanks. 💜" if result == "resisted"
             else "Unlocked. Enjoy, and stay mindful of your time.")


<<<<<<< HEAD
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
=======
def finish_nudge():
    S.stage = None
    S.msg = "Nice. Take care of yourself. 💜"


def collect_badges():
    """Badges earned during the check-in that just ended."""
    before = S.get("badges_before")
    if before is None:
        return
    S.badges_before = None
    new = sorted(earned_names(S.data) - before)
    if new:
        S.new_badges, S.badge_until = new, time.time() + 8


def end_welcome():
    S.flow = None


def jump():
    """Demo only: fast-forward the clock to the agent's next check-in and remember when it is."""
    cur = now()
    base = cur.replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % 5)
    for step in range(5, 48 * 60, 5):
        t = base + timedelta(minutes=step)
        due = due_moment(t, S.data)
        if due:
            skipped = int((t - cur).total_seconds() // 60) + 1
            S.data["offset_min"] = S.data.get("offset_min", 0) + skipped
            save(S.data)
            S.last_key = None
            name = {"morning": "wake-up", "bed": "bedtime", "habit": "habit"}[due[0]]
            S.jump_note = (f"Fast-forwarded {skipped // 60} h {skipped % 60} min to the {name} check-in "
                           f"at {fmt_time(now())}")
            S.jump_until = time.time() + 25
            return
    S.jump_note, S.jump_until = "No check-in found in the next 48 hours.", time.time() + 10


def set_time():
    """Demo only: fast-forward the agent's clock to a chosen time of day (tomorrow if it has passed)."""
    t = S.get("demo_time")
    if t is None:
        return
    cur = now()
    goal = cur.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if goal <= cur:
        goal += timedelta(days=1)
    minutes = int((goal - cur).total_seconds() // 60) + 1
    S.data["offset_min"] = S.data.get("offset_min", 0) + minutes
    save(S.data)
    S.last_key = None
    S.jump_note = f"Fast-forwarded {minutes // 60} h {minutes % 60} min to {fmt_time(now())}"
    S.jump_until = time.time() + 25


def skip_days():
    S.data["skip_days"] += int(S.get("demo_days", 30))
    save(S.data)


def stop_using():
    if os.path.exists(data_path()):
        os.remove(data_path())
    mt.delete_user(S.uid)
    S.logged_seed = None
    st.cache_resource.clear()
    S.data = new_data()
    S.stage, S.flow, S.started, S.stopped, S.page = None, None, False, True, "main"


def restart():
    S.stopped = False


def reset_all():
    stop_using()
    S.stopped = False


# --- survey (one question at a time) ---
def start_survey(steps, after):
    p = S.data["profile"]
    S.steps, S.si, S.draft, S.after, S.flow = steps, 0, {}, after, "survey"
    S.ms_goals = list(p["goals"]) if p else []
    S.ms_hobbies = list(p["hobbies"]) if p else []
    S.oth_goals = S.oth_hobbies = ""
    for k in ("ss", "se", "bed", "wake"):
        S["h_" + k] = HOURS[p[k] if p else DEFAULT_HOUR[k]]


def begin():
    S.started = True
    start_survey(["goals", "hobbies", "ss", "se", "bed", "wake"], "new")


def resolve(text, table, labels):
    """Match typed text to one of the table keys, by key or by formal label."""
    t = text.strip().lower()
    for k in table:
        if t in (k, labels[k].lower()):
            return k
    return None


def add_pick(key, okey, value):
    S[key] = list(S[key]) + [value]
    S[okey] = ""


def next_step(step):
    if step in ("goals", "hobbies"):
        table, labels = (GOALS, GOAL_LABELS) if step == "goals" else (HOBBIES, HOBBY_LABELS)
        items = list(S["ms_" + step])
        match = resolve(S["oth_" + step], table, labels)
        if match and match not in items:
            items.append(match)
        S.draft[step] = items
    else:
        S.draft[step] = HOURS.index(S["h_" + step])
    S.si += 1
    if S.si >= len(S.steps):
        finish_survey()


def finish_survey():
    d = S.draft
    if S.after == "new":
        S.data["profile"] = {k: d[k] for k in ("goals", "hobbies", "ss", "se", "bed", "wake")}
        S.data.update(day0=time.time(), skip_days=0, next_monthly=30)
        S.flow = "welcome"
    else:
        S.data["profile"].update(goals=d["goals"], hobbies=d["hobbies"])
        S.flow, S.mq = "monthly", "tune"
    save(S.data)
    st.cache_resource.clear()


# --- monthly check-in (one question at a time) ---
def rate(n):
    S.mq = "stop" if n == 1 else "change"


def go_mq(step):
    S.mq = step


def change_goals():
    start_survey(["goals", "hobbies"], "monthly")


def tune(choice):
    if choice == "lighter":
        S.data["threshold"] = 0.8      # only block the strongest patterns
    elif choice == "heavier":
        S.data["threshold"] = 0.5      # block more patterns
    S.data["next_monthly"] += 30
    save(S.data)
    st.cache_resource.clear()
    S.flow, S.mq = None, None


def confetti_html():
    colors = ["#2F4F3A", "#9DBB9A", "#E8A0AC", "#F5D9DC", "#c9a24b"]
    bits = "".join(
        f'<i style="--x:{random.randint(-130, 130)}px;--y:{random.randint(40, 130)}px;'
        f'--r:{random.randint(-300, 300)}deg;background:{random.choice(colors)};'
        f'animation-delay:{random.random() * 0.3:.2f}s"></i>' for _ in range(26))
    return f'<div class="confetti">{bits}</div>'


def say(text):
    st.markdown(f'<div class="say">{html.escape(text, quote=False)}</div>', unsafe_allow_html=True)


def brand():
    st.markdown('<div class="brand">Mindful Scroll</div>', unsafe_allow_html=True)


def sound_player():
    components.html(SOUND_HTML, height=110)


@st.fragment(run_every=10)
def watcher():
    """Every 10 seconds the agent looks at the clock and decides whether to check in."""
    due = due_moment(now(), S.data)
    if due:
        kind, app = due
        key = (now().strftime("%Y-%m-%d %H"), kind)
        if S.last_key != key:
            S.last_key, S.kind, S.target_app = key, kind, app
            S.activity = pick_activity(S.data["profile"], kind)
            S.breathed = False
            S.stage = "notify"
            st.rerun()
>>>>>>> 6c44468 (Mindful Scroll: final hackathon version)


@st.fragment(run_every=1)
def countdown():
    left = S.block_end - time.time()
    if left <= 0:
        S.stage = "choice"
        st.rerun()
        return
    st.progress(min(1.0, left / S.block_total))
<<<<<<< HEAD
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
=======
    st.write(f"Blocked for {int(left) + 1} more seconds (demo speed)")


@st.fragment(run_every=1)
def breath_watch():
    """Waits (without blocking the page) until the 5 breaths are done, then moves on."""
    if time.time() >= S.breathe_end:
        S.badges_before = S.badges_before if S.badges_before is not None else earned_names(S.data)
        S.data["breaths"] += 1
        log("breathing_break", BREATH_SECONDS / 60, S.kind or "")
        S.data["mindful_min"] += BREATH_SECONDS / 60
        S.breathed = True
        save(S.data)
        go_activity()
        st.rerun()


@st.fragment(run_every=1)
def badge_timer():
    if time.time() >= S.badge_until:
        go_page("trophies")
        st.rerun()


def blocked_panel():
    if S.target_app is None:
        chips = "".join(f'<span class="chip lock">🔒 {a}</span>' for a in APPS)
        note = "Your social apps are paused for this focus break."
    else:
        chips = "".join(f'<span class="chip{" lock" if a == S.target_app else ""}">'
                        f'{"🔒 " if a == S.target_app else ""}{a}</span>' for a in APPS)
        note = f"{S.target_app} is blocked right now."
    st.markdown(f'<div class="judge">{chips}<br><small>Judge view: {note} '
                'This demo shows the block; the phone version will lock the real app.</small></div>',
                unsafe_allow_html=True)


def next_checkin(n):
    """The agent's next planned check-in after time n: (datetime, kind) or None."""
    base = n.replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % 5)
    for step in range(5, 48 * 60, 5):
        t = base + timedelta(minutes=step)
        due = due_moment(t, S.data)
        if due:
            return t, due[0]
    return None


@st.fragment(run_every=2)
def clock_corner():
    """Time bar in the corner: the agent's clock, the day, and when the next check-in will happen."""
    n, p = now(), S.data["profile"]
    key = n.strftime("%Y-%m-%d %H:%M")
    if S.nc_key != key:
        S.nc_key, S.nc_val = key, next_checkin(n)
    nxt = S.nc_val
    pct = lambda h: f"{h / 24 * 100:.1f}"
    school = ""
    if n.weekday() < 5 and p["se"] > p["ss"]:
        school = f'<span class="sch" style="left:{pct(p["ss"])}%;width:{pct(p["se"] - p["ss"])}%"></span>'
    marks = "".join(f'<i class="mk" style="left:{pct(h)}%"></i>' for h in (p["wake"], p["bed"]))
    tick = ""
    text = "No check-in planned"
    if nxt:
        t, kind = nxt
        name = {"morning": "wake-up", "bed": "bedtime", "habit": "habit"}[kind]
        day = "" if t.date() == n.date() else t.strftime("%a ")
        text = f"Next check-in: {day}{fmt_time(t)} ({name})"
        if t.date() == n.date():
            tick = f'<i class="nx" style="left:{pct(t.hour + t.minute / 60)}%"></i>'
    now_pct = pct(n.hour + n.minute / 60)
    jump = f'<div class="jn">⏩ {html.escape(S.jump_note)}</div>' if S.jump_note and time.time() < S.jump_until else ""
    st.markdown(f'<div class="clock"><div class="t">{n.strftime("%a %b")} {n.day} · {fmt_time(n)}</div>'
                f'<div class="trk">{school}{marks}{tick}<i class="nw" style="left:{now_pct}%"></i></div>'
                f'<div>🌅 {HOURS[p["wake"]]} · 🌙 {HOURS[p["bed"]]}</div>'
                f'<div class="nxt">{text}</div>{jump}</div>', unsafe_allow_html=True)


def new_sim():
    S.sim_seed, S.logged_seed = random.randrange(2 ** 31), None
    S.agent_text = S.agent_report = None


def extract_cb():
    """Free text -> strict JSON record -> SQLite. Uses the AI service when available."""
    text = S.get("log_text", "").strip()
    if not text:
        return
    factory = None
    if LLM_ON:
        import anthropic
        factory = lambda: anthropic.Anthropic(timeout=8.0, max_retries=0)
    try:
        n = now()
        rec = mt.extract_from_text(text, n.date(), n.hour, factory, LLM_MODEL)
        mt.log_event(S.uid, rec)
        S.log_result = rec
    except Exception as e:
        S.log_result = {"error": str(e)}


def judge_tool(p):
    """Random, schedule-based screen time, next to what the agent found. For judges."""
    usage, rules = get_usage(p)
    with st.expander("Judge tool: simulated screen time", expanded=True):
        st.caption(f"Random every time the site opens, built from this schedule: school {HOURS[p['ss']]} to "
                   f"{HOURS[p['se']]}, wake {HOURS[p['wake']]}, bed {HOURS[p['bed']]}. It is saved in the "
                   "local SQLite file and the agent learns from it.")
        st.button("Generate a new random week", key="judge_new", on_click=new_sim)
        daily = usage.groupby("date")["minutes"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Per day (7 days)", f"{daily.tail(7).mean() / 60:.1f} h")
        c2.metric("Top app", usage.groupby("app")["minutes"].sum().idxmax())
        st.write("**Minutes per day, last 14 days**")
        st.bar_chart(daily.tail(14), color="#2F4F3A")
        st.write("**Average minutes by hour of day**")
        st.bar_chart((usage.groupby("hour")["minutes"].sum() / max(1, daily.size)).reindex(range(24), fill_value=0),
                     color="#9DBB9A")
        last7 = sorted(usage["date"].unique())[-7:]
        st.write("**Last 7 days by app (minutes)**")
        st.dataframe(usage[usage["date"].isin(last7)].pivot_table(
            index="date", columns="app", values="minutes", aggfunc="sum", fill_value=0))

        st.write("**Answer key: habits planted in this data**")
        for r in rules:
            st.markdown(f"• **{r['app']}** around **{HOURS[r['hour']]}** {r['when']} "
                        f"({r['chance']:.0%} of the time)")
        rep = get_model(p)
        st.write("**Patterns the agent found**")
        for sentence in top_patterns(rep["forecast"], eff_threshold(rep), limit=8):
            st.markdown("• " + sentence)
        st.caption(f"Tested on hours it had not trained on: accuracy {rep['accuracy']:.0%}. When it was at least "
                   f"{rep['t_star']:.0%} sure of a habit, it was right {rep['precision']:.0%} of the time.")

        if st.button("Ask the agent to explain my patterns", key="judge_explain"):
            with st.spinner("Running the pandas tool…"):
                factory = None
                if LLM_ON:
                    import anthropic
                    factory = lambda: anthropic.Anthropic(timeout=20.0, max_retries=0)
                S.agent_text, S.agent_report = mt.agent_explain(S.uid, p, factory, LLM_MODEL)
        if S.agent_text:
            st.write(S.agent_text)
            st.caption("Pandas did the maths (below). The AI only explains these numbers."
                       if LLM_ON else "No AI key set, so this is a simple template. The numbers are real.")
            st.json(S.agent_report, expanded=False)

        st.write("**Saved data (strict schema)**")
        st.text_input("Log something in your own words", key="log_text",
                      placeholder="e.g. I scrolled TikTok for 40 minutes at 4pm")
        st.button("Extract and save", key="judge_extract", on_click=extract_cb)
        if S.log_result:
            st.json(S.log_result)
        st.dataframe(mt.recent_events(S.uid, 8))
        if st.checkbox("Show the JSON schema", key="judge_schema"):
            st.json(mt.SCHEMA)
        st.download_button("Download my data (CSV)", mt.export_csv(S.uid), "mindful_data.csv", "text/csv",
                           key="judge_csv")


def sidebar():
    """Trophy room link and demo tools."""
    p = S.data["profile"]
    learning = day_number(S.data) < LEARNING_DAYS
    with st.sidebar:
        st.markdown("### Menu")
        st.button("Home", key="nav_home", on_click=go_page, args=("main",))
        st.button("Trophy room", key="nav_trophy", on_click=go_page, args=("trophies",))
        if not learning:
            rep = get_model(p)
            with st.expander("Your top patterns"):
                for sentence in top_patterns(rep["forecast"], eff_threshold(rep)):
                    st.markdown("• " + sentence)
                st.caption("These are patterns, not certainties.")
        with st.expander("Demo tools (for judges)", expanded=True):
            st.caption("Demo: tracking data is simulated, and a website can't lock real phone apps, "
                       "see other apps, or run in the background. A real version needs a native phone app.")
            st.button("Fast-forward to the next check-in", key="demo_ff", on_click=jump)
            if S.jump_note and time.time() < S.jump_until:
                st.caption("⏩ " + S.jump_note)
            st.time_input("Fast-forward to a time of day", value=datetime.strptime("08:00", "%H:%M").time(),
                          key="demo_time", step=300)
            st.button("Go to this time", key="demo_time_go", on_click=set_time)
            st.number_input("Days to fast-forward", min_value=1, max_value=365, value=30, step=1,
                            key="demo_days")
            st.button("Fast-forward", key="demo_skip", on_click=skip_days)
            st.caption(f"Now on day {int(day_number(S.data)) + 1}. "
                       f"Predictions begin on day {LEARNING_DAYS + 1}. "
                       f"Agent clock: {fmt_time(now())} (real time plus {S.data['skip_days']} days and {S.data.get('offset_min', 0)} min).")
            st.button("Reset everything", key="demo_reset", on_click=reset_all)
            if st.checkbox("Show model insights", key="demo_insights"):
                rep = get_model(p)
                st.caption(f"Accuracy {rep['accuracy']:.0%} (goal: {mt.MIN_ACCURACY:.0%}). "
                           f"When it was at least {rep['t_star']:.0%} sure, it was right {rep['precision']:.0%} "
                           f"of the time and caught {rep['recall']:.0%} of the hours an app was really used. "
                           + ("Habit notifications are on." if rep["ready"]
                              else "Not accurate enough yet, so only mindful check-ins are sent."))
                st.write("**Feature importance**")
                ranked = pd.Series(rep["importances"]).sort_values(ascending=False)
                st.bar_chart(pd.DataFrame({"importance": ranked}), color="#2F4F3A")
                top3 = ", ".join(f"{n} ({v:.0%})" for n, v in ranked.head(3).items())
                st.caption(f"Strongest predictors: {top3}. The data is simulated from schedule-based "
                           "habits, so this shows the model recovering those patterns.")
        judge_tool(p)


# ---------- screens ----------
def survey_flow():
    brand()
    step = S.steps[S.si]
    st.progress(S.si / len(S.steps))
    if step in ("goals", "hobbies"):
        table, labels = (GOALS, GOAL_LABELS) if step == "goals" else (HOBBIES, HOBBY_LABELS)
        key, okey = "ms_" + step, "oth_" + step
        st.subheader("Which goals would you like to pursue?" if step == "goals"
                     else "Which activities do you enjoy?")
        picks = st.multiselect("Select all that apply", list(table), key=key,
                               format_func=lambda k: labels[k],
                               placeholder="Select one or more options")
        raw = st.text_input("Other (optional)", key=okey)
        match = resolve(raw, table, labels)
        unmatched = raw.strip() != "" and match is None
        if unmatched:
            options = [k for k in table if k not in picks]
            by_label = {labels[k].lower(): k for k in options}
            close = difflib.get_close_matches(raw.strip().lower(), list(by_label), n=3, cutoff=0.4)
            ideas = [by_label[c] for c in close] or options[:3]
            st.write(f"We do not yet have activity suggestions for “{raw.strip()}”. "
                     "Would you like to choose one of these instead?")
            for k in ideas:
                st.button("Add: " + labels[k], key="s_" + step + k, on_click=add_pick, args=(key, okey, k))
        ready = (len(picks) > 0 or match is not None) and not unmatched
        st.button("Continue", on_click=next_step, args=(step,), disabled=not ready)
    else:
        question = {"ss": "At what time does your school day begin?",
                    "se": "At what time does your school day end?",
                    "bed": "At what time do you usually go to bed?",
                    "wake": "At what time do you usually wake up?"}[step]
        st.subheader(question)
        st.selectbox(question, HOURS, key="h_" + step, label_visibility="collapsed")
        st.button("Continue", on_click=next_step, args=(step,))


def welcome_flow():
    brand()
    st.title("Getting to know you 🌱")
    st.write(f"For the first six weeks, I’m learning your routine. "
             "I’ll check in after you wake up and before bed.")
    st.write("")
    st.button("Continue", key="big_yes", on_click=end_welcome)


def monthly_flow():
    brand()
    st.subheader("Monthly check-in 🗓️")
    if S.mq == "rate":
        st.write("How have you been liking Mindful Scroll?")
        cols = st.columns(5)
        for i in range(5):
            cols[i].button(str(i + 1), key=f"r{i}", on_click=rate, args=(i + 1,))
        st.caption("1 = not at all, 5 = love it")
    elif S.mq == "stop":
        st.write("I’m sorry it hasn’t been working for you. Would you like to stop using Mindful Scroll?")
        st.button("Yes, stop and delete my data", on_click=stop_using)
        st.button("No, keep going", on_click=go_mq, args=("change",))
    elif S.mq == "change":
        goals = ", ".join(GOAL_LABELS.get(g, g) for g in profile["goals"])
        hobbies = ", ".join(HOBBY_LABELS.get(h, h) for h in profile["hobbies"])
        st.write(f"Your goals: **{goals}**. Your activities: **{hobbies}**. Do you want to change them?")
        st.button("Yes, change them", on_click=change_goals)
        st.button("No, they’re right", on_click=go_mq, args=("tune",))
    elif S.mq == "tune":
        st.write("How should I handle app blocking?")
        st.button("A little lighter (only my strongest hunches)", on_click=tune, args=("lighter",))
        st.button("Keep it as is", on_click=tune, args=("same",))
        st.button("A little heavier (more patterns)", on_click=tune, args=("heavier",))


def trophy_room():
    brand()
    st.title("Trophy room 🏆")
    hist = data["hist"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Scrolls skipped", hist.count("resisted"))
    c2.metric("Best streak", best_streak(hist))
    c3.metric("Breathing breaks", data["breaths"])
    badges = badge_list(data)
    cards = ""
    for icon, name, desc, have, need in badges:
        got = have >= need
        sub = desc if got else f"{desc} ({min(have, need)}/{need})"
        cls = "trophy" + (" new" if got and name in S.just_earned else "")
        cards += (f'<div class="{cls}"><div class="medal sm{"" if got else " lock"}">'
                  f'<div class="in">{icon if got else "🔒"}</div></div>'
                  f'<b>{name}</b><br><small>{sub}</small></div>')
    st.markdown(f'<div class="tgrid">{cards}</div>', unsafe_allow_html=True)
    st.caption(f"{sum(b[3] >= b[4] for b in badges)} of {len(badges)} badges unlocked")
    earned = [b for b in badges if b[3] >= b[4]]
    if earned:
        st.subheader("Share a badge")
        pick = st.selectbox("Choose a badge", [b[1] for b in earned], key="share_pick")
        icon, name, desc = next((b[0], b[1], b[2]) for b in earned if b[1] == pick)
        text = f"I just earned the “{name}” badge on Mindful Scroll {icon} ({desc})."
        if APP_URL:
            text += " " + APP_URL
        st.code(text, language=None)        # tap the copy icon, then paste it to a friend
        q = urllib.parse.quote(text)
        st.markdown(f"[Share on WhatsApp](https://wa.me/?text={q}) · "
                    f"[Share by email](mailto:?subject=My%20Mindful%20Scroll%20badge&body={q})")
        st.caption("This shares only the badge name, nothing about your habits or data.")
    st.button("Back home", key="big_yes", on_click=go_page, args=("home",))


def day_banner(day, learning):
    if not learning:                      # after the 42 learning days: just the day number
        st.markdown(f'<div class="day-card"><div class="leaf">🌿</div><div class="big">Day {day}</div></div>',
                    unsafe_allow_html=True)
        return
    week = (day - 1) // 7 + 1
    pos = (day - 1) % 7
    dots = "".join(f'<span class="dot {"done" if i < pos else "now" if i == pos else ""}"></span>'
                   for i in range(7))
    pct = min(100, (day - 1) / LEARNING_DAYS * 100)
    st.markdown(f'<div class="day-card"><div class="leaf">🌿</div>'
                f'<div class="big">Day {day} of {LEARNING_DAYS}</div>'
                f'<div class="sub">Week {week} of {LEARNING_DAYS // 7}</div><div class="dots">{dots}</div>'
                f'<div class="bar"><div style="width:{pct:.0f}%"></div></div></div>',
                unsafe_allow_html=True)


if S.stopped:
    st.title("Take care 💜")
    st.write("Your data has been deleted and Mindful Scroll has stopped. You can start again whenever you want.")
    st.button("Start again", on_click=restart)
    st.stop()
if not S.started:
    st.markdown('<div class="hero"><div class="orb o1"></div><div class="orb o2"></div>'
                '<div class="leaf">🌿</div><div class="ttl">Mindful Scroll</div>'
                '<div class="tag">A gentle check-in for your scrolling habits</div>'
                '<div class="pills"><span>🌱 Notice patterns</span><span>🌬️ Breathe</span>'
                '<span>💜 Choose calmly</span></div></div>', unsafe_allow_html=True)
    st.button("Get started", key="big_yes", on_click=begin)
    st.stop()
if S.flow == "survey":
    survey_flow()
    st.stop()
if profile and S.stage is None:
    sidebar()
if S.flow == "welcome":
    welcome_flow()
    st.stop()
if S.flow is None and day_number(data) >= data["next_monthly"]:
    S.flow, S.mq = "monthly", "rate"
if S.flow == "monthly":
    monthly_flow()
    st.stop()
if S.page == "trophies" and S.stage is None:
    trophy_room()
    st.stop()

# ---------- main screen ----------
if S.stage != "notify":
    brand()
learning = day_number(data) < LEARNING_DAYS

if S.stage in (None, "notify"):
    clock_corner()

if S.stage is None:
    collect_badges()
    if S.msg:
        st.markdown(f'<div class="msg">{S.msg}</div>', unsafe_allow_html=True)
        S.msg = None
    if S.celebrate:
        st.markdown(confetti_html(), unsafe_allow_html=True)
        S.celebrate = False
    if S.new_badges:
        info = {b[1]: b for b in badge_list(data)}
        for n in S.new_badges:
            if n not in info:
                continue
            icon, name, desc, have, need = info[n]
            st.markdown('<div class="label">New badge unlocked</div>', unsafe_allow_html=True)
            with st.container(key="badge_wrap"):
                st.markdown(f'<div class="badge"><div class="ribbon"><i></i><i></i></div>'
                            f'<div class="medal"><div class="in">{icon}</div></div>'
                            f'<div class="bname">{name}</div><div class="bdesc">{desc}</div></div>',
                            unsafe_allow_html=True)
                st.button("Open trophy room", key="badge_open", on_click=go_page, args=("trophies",))
            st.caption("Tap your badge to see it in the trophy room.")

    day = int(day_number(data)) + 1
    day_banner(day, learning)

    label, text = PROMPTS[(int(day_number(data)) * 3 + now().hour // 8 + S.prompt_shift) % len(PROMPTS)]
    st.markdown(f'<div class="label">{label}</div><div class="prompt">{html.escape(text)}</div>',
                unsafe_allow_html=True)
    st.button("Another one ↻", key="quiet_prompt", on_click=lambda: S.update(prompt_shift=S.prompt_shift + 1))
    st.markdown(f'<div class="minutes"><div class="n">{int(round(data["mindful_min"]))}</div>'
                '<div class="l">mindful minutes</div>'
                '<div class="l" style="font-size:.72rem;opacity:.55;margin-top:4px">Breathing time plus time you stayed '
                'off blocked apps, measured by the app’s clock (demo speed: 1 minute = 3 seconds).</div></div>',
                unsafe_allow_html=True)
    st.button("Focus now", key="focus_now", on_click=focus_now)
    st.markdown('<div class="focus-note">Feeling overwhelmed by social media? Tap Focus now for a mindful '
                'pause: your social apps are paused, you can breathe, and you get a calmer idea to try.</div>',
                unsafe_allow_html=True)
    st.button("🏆 Trophy room", key="home_trophy", on_click=go_page, args=("trophies",))
    streak = current_streak(data["hist"])
    if streak >= 3:
        st.markdown(f'<div class="label" style="margin-top:1.5rem">🔥 {streak} skips in a row</div>',
                    unsafe_allow_html=True)
    watcher()

elif S.stage == "notify":
    st.markdown(f'<div class="notif"><b>🌿 Mindful Scroll</b> · {fmt_time(now())}<br>{NOTIF[S.kind]}</div>',
                unsafe_allow_html=True)
    st.button(YES[S.kind], key="notif_yes", on_click=start_intro)
    st.button("Not now", key="notif_no", on_click=set_stage, args=(None,))

elif S.stage == "intro":
    say(INTRO[S.kind])
    st.text_area("How are you feeling?", key="feel_text", height=110, label_visibility="collapsed",
                 placeholder="In a few words (optional), e.g. nervous about tomorrow’s test")
    st.caption("Your words are sent to an AI service to write a reply, and this app does not save them."
               if LLM_ON else "Your words are used to write a reply, and this app does not save them.")
    st.button("Share", key="big_yes", on_click=share_feeling)
    st.button("I’d rather not say", key="big_no", on_click=set_stage, args=("offer",))

elif S.stage == "reflecting":
    with st.spinner("Listening…"):
        S.reply = llm_reply(S.pending) or local_reply(S.pending, S.mood)
    S.pending = None
    S.stage = "offer"
    st.rerun()

elif S.stage == "support":
    st.title("You’re not alone 💜")
    st.write("I’m really glad you told me. I’m an app, not a counselor, so please reach out to a real person right now.")
    st.markdown("- **Call or text 988** (Suicide & Crisis Lifeline, US): [Call 988](tel:988) · [Text 988](sms:988)\n"
                "- **Text HOME to 741741** (Crisis Text Line, US)\n"
                "- **In danger right now?** Call 911 (or your local emergency number)\n"
                "- **Outside the US?** [findahelpline.com](https://findahelpline.com) lists local lines\n"
                "- Tell a parent, a teacher, a school counselor, or another adult you trust")
    st.write("**While you reach out:** name 5 things you can see, 4 you can touch, 3 you can hear, "
             "2 you can smell, and 1 you can taste.")
    st.button("I’m okay for now, go back", key="big_yes", on_click=set_stage, args=(None,))

elif S.stage == "offer":
    offer_breathing = data["declines"] < 2 or S.mood == "negative"
    if not offer_breathing and not S.reply:
        go_activity()
        st.rerun()
    if S.reply:
        safe = html.escape(S.reply).replace("\n", "<br>")
        st.markdown(f'<div class="msg">{safe}</div>', unsafe_allow_html=True)
    if offer_breathing:
        st.button("Yes, let’s breathe 🌬️", key="big_yes", on_click=start_breathing)
        st.button("Not now", key="big_no", on_click=decline_breathing)
    else:
        st.button("Continue", key="big_yes", on_click=go_activity)

elif S.stage == "breathe":
    with st.spinner("Preparing your guided voice…"):
        clips = voice_clips()
    S.breathe_end = time.time() + BREATH_SECONDS + 2.5     # start the clock once the voice is ready
    components.html(BREATHE_HTML.replace("__CLIPS__", json.dumps(clips)), height=440)
    st.button("End early", key="quiet_end", on_click=end_breathing_early)
    breath_watch()

elif S.stage == "activity":
    idea, start, _ = S.activity
    say("Here’s a gentle idea: " + idea)
    st.write("**How to start:** " + start)
    st.button("Done 💜", key="big_yes", on_click=finish_nudge)

elif S.stage == "block":
    sound_player()
    idea, start, _ = S.activity
    say("A gentle idea for right now: " + idea)
    st.write("**How to start:** " + start)
    blocked_panel()
    countdown()
    st.button("I’m good, I’ll skip scrolling ✅", key="big_yes", on_click=finish_habit, args=("resisted",))

elif S.stage == "choice":
    say("Nice job taking a break. It’s your choice now. 💜")
    st.button("I’ll skip scrolling ✅", key="big_yes", on_click=finish_habit, args=("resisted",))
    st.button("Unlock the app", key="big_no", on_click=finish_habit, args=("opened",))
>>>>>>> 6c44468 (Mindful Scroll: final hackathon version)
