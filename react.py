"""
The decide step of the ReAct loop.

Before a persona writes anything, we ask it what it wants to do.
Only then do we generate the content for whatever it picked.

Two reasons for splitting it:
  1. If you ask the model to decide AND write in one go, it always writes.
     Separating them makes "do nothing" a real option.
  2. You can print the decision before any content exists, so when behaviour
     looks wrong you can tell whether the decision or the writing was at fault.
"""

from llm import chat_json

# Which action each "app" unlocks. A persona is only ever offered actions
# for apps listed in their persona file.
ACTION_CATALOGUE = {
    "forum": {
        "name": "reply_to_thread",
        "description": "Write a reply in the forum thread you are looking at",
    },
    "mastodon": {
        "name": "post_toot",
        "description": "Post a short public message on Mastodon (not a reply)",
    },
    "email": {
        "name": "send_email",
        "description": "Send an email to one person you know",
    },
}

# Always available. Real people ignore most things.
DO_NOTHING = {
    "name": "do_nothing",
    "description": "Do nothing right now - this does not concern you, "
                   "or you have nothing worth adding",
}


def available_actions(persona):
    """The actions this persona is allowed to take, based on their apps list."""
    actions = []
    for app in persona.get("apps", []):
        if app in ACTION_CATALOGUE:
            actions.append(ACTION_CATALOGUE[app])
    actions.append(DO_NOTHING)
    return actions


def build_decide_prompt(persona):
    """Instructions for the decide step.

    Deliberately shorter than the writing prompt - no tone, no word limits,
    no habits. Those matter when writing, not when choosing.
    """
    opinions = persona.get("opinions", {}) or {}
    goals = persona.get("goals", []) or []

    lines = [
        f"You are {persona['name']}, a {persona.get('job', 'person')} "
        f"at {persona.get('employer', 'a company')}, posting under the username "
        f"'{persona['username']}'."
    ]

    if opinions:
        lines.append("\nWhat you think:")
        for topic, view in opinions.items():
            lines.append(f"- {topic.replace('_', ' ')}: {view}")

    if goals:
        lines.append("\nWhat you want:")
        for goal in goals:
            lines.append(f"- {goal}")

    talks_to = persona.get("talks_to", {}) or {}
    if talks_to:
        lines.append("\nPeople you know:")
        for who, relationship in talks_to.items():
            lines.append(f"- {who} ({relationship})")

    lines.append(
        "\nYou are deciding what to do next, not writing anything yet. "
        "Choose the single action you would most plausibly take. It is "
        "completely normal to choose to do nothing - most people ignore most "
        "conversations. Only get involved if this actually concerns you or you "
        "have something real to add."
    )

    return "\n".join(lines)


def decide(persona, situation, model, ollama_url, temperature=0.7):
    """Ask the persona what it wants to do. Returns a dict."""
    actions = available_actions(persona)

    options = "\n".join(
        f"- {a['name']}: {a['description']}" for a in actions
    )
    valid_names = [a["name"] for a in actions]

    user_message = (
        f"Here is what is happening:\n\n{situation}\n\n"
        f"Your options:\n{options}\n\n"
        "Reply with JSON only, in this exact shape:\n"
        '{"action": "<one of the option names>", '
        '"target": "<who or what, or empty>", '
        '"reason": "<one short sentence>"}'
    )

    raw = chat_json(
        build_decide_prompt(persona),
        user_message,
        model=model,
        ollama_url=ollama_url,
        temperature=temperature,
    )

    return _clean(raw, valid_names)


def _text(value):
    """Turn whatever the model sent into a plain string.

    Models often send null for target or reason. str(None) is the word
    "None", which is not empty, so later checks like `target or "someone"`
    would never fire. Catch it here instead.
    """
    if value is None:
        return ""
    return str(value).strip()


def _clean(raw, valid_names):
    """Models go off-script. Make sure what comes back is usable."""
    if not isinstance(raw, dict):
        return {"action": "do_nothing", "target": "",
                "reason": "could not read the model's decision"}

    action = str(raw.get("action", "")).strip()
    if action not in valid_names:
        # Sometimes the model invents an action or returns a sentence.
        return {"action": "do_nothing", "target": "",
                "reason": f"invalid action '{action}', skipping"}

    return {
        "action": action,
        "target": _text(raw.get("target")),
        "reason": _text(raw.get("reason")),
    }
