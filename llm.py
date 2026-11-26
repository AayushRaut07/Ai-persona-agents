"""
Talks to Ollama.

Ollama runs a small web server on your machine (port 11434 by default).
We send it JSON and get JSON back. That is all this file does.

Nothing here knows about personas - keep it that way.
"""

import json
import time

import requests

# Seconds to wait for one generation. The lab machine has no GPU, so a
# long prompt on a CPU-only box can take minutes. 180 was not enough and
# runs died part-way through.
TIMEOUT = 600


def chat(system_prompt, user_message, model, ollama_url, temperature=0.8,
         timeout=TIMEOUT, num_predict=None):
    """Send one message to the model.

    system_prompt : who the model should be (the persona)
    user_message  : what it is reacting to (the forum post)

    Returns (reply, seconds_taken). The timing is reported in the run log,
    which is how we answered the question about latency per message.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }

    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    started = time.time()
    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    elapsed = time.time() - started

    reply = response.json()["message"]["content"].strip()
    return reply, elapsed


def chat_json(system_prompt, user_message, model, ollama_url, temperature=0.7,
              timeout=TIMEOUT, num_predict=None):
    """Like chat(), but asks Ollama for JSON and parses it.

    Returns the parsed dict only - no timing, unlike chat().

    Ollama's "format": "json" makes the model reply with valid JSON instead of
    prose. Used by the decide step, where code has to act on the answer.
    Returns a dict, or None if the reply could not be parsed.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
    }

    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    response = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    text = response.json()["message"]["content"].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def check_connection(ollama_url):
    """Returns True if Ollama is running and reachable."""
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=5).raise_for_status()
        return True
    except Exception:
        return False
