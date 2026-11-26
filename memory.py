"""
Per-persona memory.

Each persona has their own diary file:  memory/<username>.json
It records what they posted, emails they sent, and emails they received.

Before a persona decides or writes anything, the last few entries of THEIR
OWN diary are added to their prompt. They never see anyone else's file.

That one rule is the answer to "if one model plays many personas, how do you
stop one person's context leaking into another's?" - each persona only ever
gets their own memory, and the model itself remembers nothing between calls.
"""

import json
import re
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "memory"


def _path(username, memory_dir=None):
    directory = Path(memory_dir) if memory_dir is not None else MEMORY_DIR
    return directory / f"{username}.json"


def load_memory(username, memory_dir=None):
    """Everything this persona remembers, oldest first."""
    path = _path(username, memory_dir)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remember(username, kind, text, other="", topic="", memory_dir=None):
    """Add one entry to a persona's diary.

    kind  : what happened - posted, sent_email, received_email, posted_toot
    text  : the actual words
    other : the other person involved, if any
    topic : what the conversation was about, so memories don't get mixed up
    memory_dir : optional runtime directory; old callers keep the default
    """
    path = _path(username, memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_memory(username, memory_dir)
    entries.append({
        "when": datetime.now().isoformat(timespec="seconds"),
        "kind": kind,
        "with": other,
        "topic": topic,
        "text": text,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def recall(username, limit=6, max_chars=300, memory_dir=None):
    """The recent part of a persona's diary, as text for a prompt.

    Only the last few entries, and each one trimmed, so the prompt stays
    short and the model stays fast.
    """
    entries = load_memory(username, memory_dir)[-limit:]
    if not entries:
        return ""

    labels = {
        "posted": "You posted on the forum",
        "posted_toot": "You posted on Mastodon",
        "sent_email": "You emailed {other}",
        "received_email": "You received an email from {other}",
    }

    lines = ["Things you remember, oldest first. Each one is something YOU said "
             "or received - do not confuse your own words with other people's. "
             "Use these to stay consistent, but do NOT repeat them word for word; "
             "real people don't copy their own old messages:"]
    for e in entries:
        label = labels.get(e["kind"], e["kind"]).format(other=e.get("with", ""))
        if e.get("topic"):
            label += f' (in a conversation about "{e["topic"]}")'
        text = e["text"].replace("\n", " ")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f'- {label}: "{text}"')
    return "\n".join(lines)


def forget_everyone():
    """Wipe all memory files. Used to start a fresh simulation."""
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.json"):
            f.unlink()


def resolve_recipient(target, sender_key, personas):
    """Work out which persona an email is meant for.

    The model returns a free-text target like "Marcus" or "mreeve" or
    "my colleague Marcus". Match it against every other persona's file name,
    username and first name. Returns the persona key, or None if the email
    is for someone who is not being simulated.

    Matching is on whole words. Matching on substrings looked simpler but
    picked the wrong person: "the same team" contains "sam".
    """
    words = set(re.findall(r"[a-z0-9']+", (target or "").lower()))
    if not words:
        return None

    for key, p in personas.items():
        if key == sender_key:
            continue
        first_name = p["name"].split()[0].lower()
        if key in words or p["username"].lower() in words or first_name in words:
            return key
    return None
