"""
Talks to the forum.

A small HTTP API: list threads, read a thread, start a thread, reply.
A post carries a username, so a persona posting is a POST with their
username on it.

Nothing here knows about personas, the same way llm.py does not.
"""

import requests

TIMEOUT = 15


def is_up(base_url):
    """True if the forum answers. Used to decide whether to use it at all."""
    try:
        requests.get(f"{base_url}/health", timeout=5).raise_for_status()
        return True
    except Exception:
        return False


def list_threads(base_url):
    """Every thread on the board, most recently active first.

    Only the opening posts - replies come back from read_thread.
    """
    response = requests.get(f"{base_url}/api/posts", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get("posts", [])


def read_thread(base_url, thread_id):
    """One thread as [{username, text}], opening post first.

    This is the shape the rest of the code already uses for a thread, so
    reading from the real forum and holding a thread in memory look the
    same to everything downstream.
    """
    response = requests.get(f"{base_url}/api/posts/{thread_id}", timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    posts = [{"username": data["username"], "text": data["content"]}]
    for r in data.get("replies", []):
        posts.append({"username": r["username"], "text": r["content"]})
    return posts


def start_thread(base_url, username, content):
    """Post a new thread. Returns its id."""
    response = requests.post(
        f"{base_url}/api/posts",
        json={"username": username, "content": content},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["id"]


def reply(base_url, thread_id, username, content):
    """Add a reply to a thread."""
    response = requests.post(
        f"{base_url}/api/posts/{thread_id}/replies",
        json={"username": username, "content": content},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
