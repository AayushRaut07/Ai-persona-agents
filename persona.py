"""
Loads persona files and turns them into a system prompt.

Three layers, deliberately kept separate:
  1. traits + vocabulary style - HOW they write   (reusable, mix and match)
  2. persona file              - WHO they are     (job, employer, relationships)
  3. opinions and goals        - WHAT they think  (stops everyone agreeing)

The "how they write" layer is split in two so you can combine them:
  traits/  - blunt, warm, chatty, timid   (length, emoji, tone)
  styles/  - professional, casual, formal (formality, typos)

A persona picks one or more traits and one style, so blunt + professional
and blunt + casual are two different writers without two extra files.
"""

import random
import re
from pathlib import Path

import yaml

BASE = Path(__file__).parent


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config():
    return _load_yaml(BASE / "config.yaml")


def load_persona(persona_name, folder=None):
    """Load a persona and merge in its traits and vocabulary style.

    folder lets a separate scenario keep its own people in its own folder.
    The traits and styles are always the shared ones, which is the point of
    splitting them out - a generated company reuses the same four voices.
    """
    where = Path(folder) if folder else BASE / "personas"
    persona = _load_yaml(where / f"{persona_name}.yaml")

    parts = []
    for trait in persona.get("traits", []) or []:
        parts.append(_load_yaml(BASE / "traits" / f"{trait}.yaml"))
    if persona.get("vocabulary"):
        parts.append(_load_yaml(BASE / "styles" / f"{persona['vocabulary']}.yaml"))

    persona["style"] = _merge_styles(parts)
    persona["habits"] = _pick_habits(persona)
    return persona


def _merge_styles(parts):
    """Combine the chosen files into one. Numbers average, lists join.

    One rule for everything, so it does not matter which half a setting
    lives in. Joining the habit lists is the point: blunt + professional
    and chatty + professional share some habits but not all.
    """
    merged = {}
    numbers = {}

    for part in parts:
        for key, value in part.items():
            if key == "name":
                continue
            if isinstance(value, list):
                merged.setdefault(key, []).extend(value)
            elif isinstance(value, (int, float)):
                numbers.setdefault(key, []).append(value)

    for key, values in numbers.items():
        average = sum(values) / len(values)
        if key in ("max_words", "habits_per_person"):
            merged[key] = round(average)
        else:
            merged[key] = round(average, 2)

    return merged


def _pick_habits(persona):
    """Give this persona a random subset of their combined habit pool.

    Seeded by username, so the same person always gets the same habits
    but two people sharing traits do not write identically.
    """
    style = persona.get("style", {})
    habits = style.get("habits", [])
    if not habits:
        return []

    count = min(style.get("habits_per_person", 1), len(habits))
    rng = random.Random(persona.get("username", persona.get("name", "")))
    return rng.sample(habits, count)


def _describe_formality(value):
    if value is None:
        return "neutral"
    if value >= 0.7:
        return "formal and professional"
    if value >= 0.4:
        return "neutral, workplace-appropriate"
    return "casual and relaxed"


def _describe_emoji(value):
    if value >= 0.6:
        return "- Use a few emoji naturally."
    if value >= 0.2:
        return "- Use an emoji only occasionally, at most one."
    return "- Never use emoji."


def _describe_typos(value):
    if value >= 0.6:
        return "- Make frequent small typos and informal spelling."
    if value >= 0.2:
        return "- Occasionally make a small, believable typo."
    return ""


def build_system_prompt(persona, app="forum", purpose="general"):
    """Turn a loaded persona into the instruction text sent to the model."""
    style = persona.get("style", {})
    opinions = persona.get("opinions", {}) or {}
    goals = persona.get("goals", []) or []

    lines = []

    # --- who they are ---
    lines.append(f"You are {persona['name']}. You post under the username "
                 f"'{persona['username']}' - posts from that username are your own. "
                 f"You are a {persona.get('job', 'person')} "
                 f"at {persona.get('employer', 'a company')}.")
    lines.append("Never reply to your own posts and never refer to yourself "
                 "in the third person. Address others by their first name.")

    if persona.get("background"):
        lines.append(f"Your background: {persona['background']}.")

    # An introduction thread is not the place for the cloud argument, so the
    # opinions layer is held back and replaced with introduction guidance.
    if purpose == "welcome":
        lines.append("\nYou are replying to a welcome/introduction thread. "
                     "Introduce your name, role, employer and relevant background "
                     "naturally. Mention only interests supported by your profile; "
                     "do not invent a hobby. Welcome the others in your own style. "
                     "Stay focused on introducing yourself rather than turning the "
                     "thread into an unrelated debate.")

    # --- what they think ---
    if opinions and purpose != "welcome":
        lines.append("\nYour actual views (hold these, do not sit on the fence):")
        for topic, view in opinions.items():
            lines.append(f"- {topic.replace('_', ' ')}: {view}")

    if goals:
        lines.append("\nWhat you are trying to do in this conversation:")
        for goal in goals:
            lines.append(f"- {goal}")

    # --- how they write ---
    lines.append("\nHow you write:")
    lines.append(f"- Tone: {_describe_formality(style.get('formality'))}")
    lines.append(f"- Hard limit: {style.get('max_words', 60)} words. Do not exceed it.")

    lines.append(_describe_emoji(style.get("emoji", 0.0)))

    typo_line = _describe_typos(style.get("typos", 0.0))
    if typo_line:
        lines.append(typo_line)

    for note in style.get("tone_notes", []) or []:
        lines.append(f"- {note}")

    for habit in persona.get("habits", []):
        lines.append(f"- {habit}")

    # --- channel-specific shaping ---
    # This is the answer to "do they sound the same on email vs social media".
    if app == "email":
        lines.append("\nThis is an email: use a greeting and a sign-off, "
                     "and write in full sentences.")
    elif app == "mastodon":
        lines.append("\nThis is a social media post: short, no greeting, "
                     "no sign-off, more casual than you would be at work.")
    elif purpose == "welcome":
        lines.append("\nThis is a forum introduction: a natural hello is fine, "
                     "but do not use an email subject or a formal sign-off.")
    else:
        lines.append("\nThis is a work forum reply: no greeting, no sign-off, "
                     "just the reply itself.")

    lines.append("\nStay in character. Reply only with what this person would "
                 "write - no narration, no quotation marks around it, no name label.")

    return "\n".join(lines)


def word_count_ok(text, persona):
    """Models ignore word limits in prose, so check it in code too."""
    limit = persona.get("style", {}).get("max_words", 60)
    return len(text.split()) <= limit * 1.3  # a little tolerance


def trim_to_limit(text, persona):
    """Cut a reply down to the persona's word limit.

    Asking the model to be brief does not work - every reply came back over
    the limit - so the cut happens here instead. Whole sentences are kept,
    because a reply that stops mid-sentence reads as broken rather than as
    brief. Text already short enough is returned unchanged.
    """
    limit = persona.get("style", {}).get("max_words", 60)
    words = text.split()
    if len(words) <= limit * 1.3:
        return text

    kept = []
    count = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        length = len(sentence.split())
        # Always keep the first sentence, even a long one, so nothing
        # comes back empty.
        if kept and count + length > limit:
            break
        kept.append(sentence)
        count += length

    result = " ".join(kept)

    # One very long sentence with no full stop fills `kept` on its own and
    # is still over the limit. Cut mid-sentence rather than let it through.
    if len(result.split()) > limit * 1.3:
        result = " ".join(result.split()[:limit])

    return result
