"""
Keeps personas from caving.

Small models are trained to be agreeable, so over a few turns a persona tends
to drift towards whoever it is talking to. Two defences, following the same
"ask, then check in code" pattern as the word limit:

  1. ASK   - remind the persona of their position right before they write.
  2. CHECK - after they write, ask the model one easy question: does this
             message argue FOR or AGAINST the topic? Python then compares the
             answer with the side in the persona file.

Why a for/against question instead of "is this consistent?": the first
version asked the model to judge consistency, and a 3B model is a poor judge -
it flagged Marcus for mentioning his own concerns. Sorting a message into
for / against / neutral is a much easier job, and the decision about whether
that is a problem is made in code, not by the model.

A reply only fails if it argues the OPPOSITE side. Neutral passes - being
polite or acknowledging a point is not caving.
"""

OPPOSITE = {"for": "against", "against": "for"}


def get_stance(persona):
    """(topic, side) from the persona file, or (None, None) if not set."""
    stance = persona.get("stance") or {}
    side = str(stance.get("side", "")).strip().lower()
    if side not in OPPOSITE:
        return None, None
    return stance.get("topic"), side


def position_text(persona):
    """The persona's opinions as a few plain lines."""
    opinions = persona.get("opinions", {}) or {}
    return "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in opinions.items())


def stance_reminder(persona):
    """Added to the writing instruction, right before the model writes."""
    if not persona.get("opinions"):
        return ""
    return (
        "\n\nRemember where you stand:\n"
        f"{position_text(persona)}\n"
        "You have NOT changed your mind. You can acknowledge a fair point or be "
        "friendly, but do not adopt the other person's position. Keep making "
        "your own case, and make a point you have not already made - do not "
        "reuse sentences from your earlier messages."
    )


def firm_reminder(persona):
    """Used on the retry, after a reply argued the wrong side."""
    topic, side = get_stance(persona)
    return (
        f"\n\nYour previous draft argued the wrong side. You are {side.upper()} "
        f"{topic}. Rewrite it so it clearly argues that. Keep it short and "
        "within your word limit."
    )


def which_side(reply, topic, model, ollama_url, *, chat_json_fn=None):
    """Ask the model: is this message for, against, or neutral on the topic?"""
    if chat_json_fn is None:
        # Scripts can still use the top-level llm module. Integrations pass
        # their own callback instead of modifying sys.modules.
        from llm import chat_json as chat_json_fn
    result = chat_json_fn(
        "You sort messages by which side of a debate they argue.",
        f"Topic: {topic}\n\nMessage:\n\"{reply}\"\n\n"
        "Does this message mainly argue FOR the topic, AGAINST it, or is it "
        "NEUTRAL / mixed?\n"
        'Reply with JSON only: {"side": "for" or "against" or "neutral"}',
        model=model,
        ollama_url=ollama_url,
        temperature=0.0,
    )
    if not isinstance(result, dict):
        return None
    side = str(result.get("side", "")).strip().lower()
    return side if side in ("for", "against", "neutral") else None


def holds_position(reply, persona, model, ollama_url, *, chat_json_fn=None):
    """Returns (ok, what_the_model_thought_the_reply_argued).

    Fails only when the reply argues the opposite side. If the persona has
    no stance set, or the check can't be read, it passes so the run continues.
    """
    topic, side = get_stance(persona)
    if not topic:
        return True, "no stance set"

    detected = which_side(reply, topic, model, ollama_url, chat_json_fn=chat_json_fn)
    if detected is None:
        return True, "unreadable"

    return detected != OPPOSITE[side], detected
