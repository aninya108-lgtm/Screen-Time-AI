# Mindful Scroll

Mindful Scroll is an AI agent that helps teens and young adults pause before mindless scrolling on Instagram, TikTok, Snapchat, and YouTube. It learns when you usually scroll, checks in just before those moments, and gives you a real choice about what to do next.

## What the agent does

- **Learns your routine.** You answer a short survey about your goals, hobbies, school hours, bedtime, and wake time. For the first 42 days it sends two mindful check-ins a day (an hour after you wake up and before bed) while it learns your patterns.
- **Predicts scrolling.** After 42 days, a machine learning model estimates when you're likely to open a social app. It sends a notification only when it is at least 60% sure and its measured accuracy is at least 80%.
- **Checks in with you.** It asks how you're feeling, offers optional guided breathing with calm audio, and suggests something from your own hobbies and goals, with a tip for getting started.
- **Pauses, never forces.** A short simulated block with a timer gives you a moment, then you decide whether to open the app or skip it.
- **Celebrates the pause.** Streaks, badges, and a Trophy Room reward the choice to stop, not time away from your phone.
- **Looks out for you.** If it notices crisis language, it shows a support screen. A quiet "I'm going through a really hard time" button is always available.
- **Stays in your control.** A monthly check-in lets you change your goals and hobbies or make notifications lighter or heavier.

This demo uses simulated screen-time data, because real phone usage tracking and app blocking need a native phone app. A judge demo panel in the sidebar fast-forwards time so you can see the full 42-day cycle in a few minutes.

## Run it

```
pip install -r requirements.txt
python3 -m streamlit run app.py
```

Optional extras (the app works without them):

```
pip install textblob anthropic edge-tts
```

- `textblob` improves the feelings analysis.
- `anthropic` lets an AI explain your patterns in plain words (needs an `ANTHROPIC_API_KEY`).
- `edge-tts` adds calmer voice clips for the breathing exercise.

## Files

- `app.py`: the website and the agent
- `mindful_tools.py`: the machine learning, data logging, and pandas analysis
- `requirements.txt`: the packages the app needs
pip install textblob anthropic edge-tts
textblob improves the feelings analysis.
anthropic lets an AI explain your patterns in plain words (needs an ANTHROPIC_API_KEY).
edge-tts adds calmer voice clips for the breathing exercise.
Files
app.py: the website and the agent
mindful_tools.py: the machine learning, data logging, and pandas analysis
requirements.txt: the packages the app needs
