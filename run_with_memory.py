"""
The ReAct loop, with memory, a working inbox and the stance check.

Runs everyone in one process: the hand-written personas and any generated
by make_people.py. There is nothing to start or stop between them.

Runs forever, the way real users do. Each configured day starts a fresh
thread; once those run out the personas keep talking on the last one.
They wait between actions and go away for longer stretches, so the forum
does not fill up in one burst.

Emails actually arrive: when Priya emails Marcus, it is written into
Marcus's memory as a received email, and he sees it next time he acts.

After it runs, open the memory/ folder to see exactly what each persona
remembers. Each persona only ever reads their own file.

Run it:  python run_with_memory.py
"""

import random
import time
from pathlib import Path

import yaml

import forum
from llm import chat, check_connection
from memory import forget_everyone, recall, remember, resolve_recipient
from persona import (build_system_prompt, load_config, load_persona,
                     trim_to_limit, word_count_ok)
from react import decide
from stance import firm_reminder, holds_position, stance_reminder

# The personas written by hand, in personas/. Generated ones are picked up
# from company/personas/ automatically, so this list does not change when
# you generate more people.
HAND_WRITTEN = ["marcus", "priya"]

# Where make_people.py puts the people it generates.
COMPANY = Path(__file__).parent / "company"

# How many people look at the forum in one round. A real board has a few
# regulars and a long tail who read without posting, so not everyone acts
# every time - and a CPU-only machine could not generate a reply for
# seventeen people in a round anyway.
READERS_PER_VISIT = 4

# Start every run with empty memories. Set to False to let memory build up
# across several runs of this script.
RESET_MEMORY = True

# How many turns each persona takes comes from config.yaml.

# The forum's address once it is known to be reachable, or None, in which
# case the thread is kept in memory - what happens on a laptop with no lab
# around it.
LIVE_FORUM = None

# How far down the board a persona will look. The forum returns threads
# most recently active first, so the top few are what is actually going on.
# Real people scan the top of a board, not the whole archive.
BOARD_WINDOW = 5


# The posts that start each day live in their own file.
DAYS_FILE = Path(__file__).parent / "days.yaml"


def pause(low, high, reason):
    """Wait a random number of seconds.

    Real people do not reply the instant someone posts, and they do not
    reply at a fixed interval either. The randomness is the point - a fixed
    delay still looks like a script.
    """
    seconds = random.randint(low, high)
    print(f"   ...{reason} for {seconds}s\n", flush=True)
    time.sleep(seconds)


def load_people():
    """Everyone in the simulation.

    The hand-written personas, plus anything make_people.py has generated.
    Both kinds are ordinary persona files, so nothing below here needs to
    know which is which.
    """
    people = {name: load_persona(name) for name in HAND_WRITTEN}

    generated = COMPANY / "personas"
    if generated.is_dir():
        for path in sorted(generated.glob("*.yaml")):
            people[path.stem] = load_persona(path.stem, folder=generated)

    return people


def load_days():
    """The starting posts, from days.yaml and company/days.yaml.

    Empty means the personas start their own conversations.
    """
    days = []
    for path in (DAYS_FILE, COMPANY / "days.yaml"):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            days.extend(yaml.safe_load(f) or [])
    return days



# How many of the most recent posts a persona is shown. Without a cap the
# prompt grows with every reply, generation gets slower each turn, and long
# runs eventually time out. Real people don't re-read a whole long thread
# before replying either.
THREAD_WINDOW = 8


def render_thread(thread):
    if not thread:
        return "The forum is empty - nobody has posted yet."
    recent = thread[-THREAD_WINDOW:]
    return "\n".join(f"{p['username']}: {p['text']}" for p in recent)


def with_memory(persona, text):
    """Put this persona's own memories in front of whatever they are shown."""
    memories = recall(persona["username"])
    return f"{memories}\n\n{text}" if memories else text


def perform(key, persona, personas, choice, context, topic, config, thread=None,
            thread_id=None):
    """Carry out a decision: write the content, check the stance, record it.

    context : what the persona is reacting to (a thread, or a note about their inbox)
    topic   : short description stored with the memory, so memories don't get mixed up
    thread  : the forum thread list, if this is a forum reply
    thread_id : which thread on the real forum the reply belongs to
    Returns the text written, or None.
    """
    model = persona.get("model") or config["default_model"]

    if choice["action"] == "reply_to_thread":
        app = "forum"
        if thread:
            instruction = (f"Here is the forum thread:\n\n{context}\n\n"
                           "Write your reply. If anything you remember is relevant, "
                           "you may refer back to it naturally.")
        else:
            # Nothing to reply to, so this becomes the thread that starts it.
            instruction = ("Nobody has posted on the forum yet. Start a thread "
                           "about something that is on your mind at work.")
    elif choice["action"] == "send_email":
        app = "email"
        instruction = (f"{context}\n\n"
                       f"Write an email to {choice['target'] or 'a colleague'}. "
                       "If you have had earlier emails with them, continue that conversation.")
    elif choice["action"] == "post_toot":
        app = "mastodon"
        instruction = (f"{context}\n\n"
                       "Write a short Mastodon post of your own about it.")
    else:
        return None

    # --- write, with the stance reminder added right before writing ---
    system_prompt = build_system_prompt(persona, app=app)
    base_message = with_memory(persona, instruction) + stance_reminder(persona)

    reply, elapsed = chat(system_prompt, base_message, model=model,
                          ollama_url=config["ollama_url"],
                          temperature=config.get("temperature", 0.8))

    # --- check they haven't switched sides; if they have, rewrite once ---
    ok, detected = holds_position(reply, persona, model, config["ollama_url"])
    if ok:
        print(f"   [stance check: argues {detected.upper()} - ok]")
    else:
        print(f"   [stance check: argues {detected.upper()} - wrong side, rewriting...]")
        reply, retry_time = chat(system_prompt, base_message + firm_reminder(persona),
                                 model=model, ollama_url=config["ollama_url"],
                                 temperature=config.get("temperature", 0.8))
        elapsed += retry_time
        ok, detected = holds_position(reply, persona, model, config["ollama_url"])
        print(f"   [after rewrite: argues {detected.upper()}]")

    # The model ignores the word limit in the prompt, so cut it here.
    reply = trim_to_limit(reply, persona)

    flag = "" if word_count_ok(reply, persona) else "  [OVER WORD LIMIT]"
    print(f"{persona['username']} ({app}, {elapsed:.1f}s){flag}:")
    print(f"{reply}\n")

    # --- record it, in the right diaries ---
    if choice["action"] == "reply_to_thread":
        if thread is not None:
            thread.append({"username": persona["username"], "text": reply})
        if LIVE_FORUM and thread_id:
            try:
                forum.reply(LIVE_FORUM, thread_id, persona["username"], reply)
                print("   (posted to the forum)\n")
            except Exception as problem:
                # A forum that goes down should not kill a long run.
                print(f"   (could not reach the forum: {problem})\n")
        remember(persona["username"], "posted", reply, topic=topic)

    elif choice["action"] == "post_toot":
        remember(persona["username"], "posted_toot", reply, topic=topic)

    elif choice["action"] == "send_email":
        recipient_key = resolve_recipient(choice["target"], key, personas)
        if recipient_key:
            recipient = personas[recipient_key]
            remember(persona["username"], "sent_email", reply,
                     other=recipient["name"].split()[0], topic=topic)
            remember(recipient["username"], "received_email", reply,
                     other=persona["name"].split()[0], topic=topic)
            print(f"   (delivered to {recipient['username']}'s inbox)\n")
        else:
            remember(persona["username"], "sent_email", reply,
                     other=choice["target"] or "someone", topic=topic)
            print("   (recipient isn't simulated - kept in sender's memory only)\n")

    return reply


def pick_thread(base_url):
    """Choose a thread to look at, from the busiest end of the board.

    The forum returns threads most recently active first, so the top few
    are whatever is going on. Picking from those rather than always the
    same one is what lets a persona notice a post somebody else made after
    the simulation started.
    """
    try:
        threads = forum.list_threads(base_url)
    except Exception:
        return None
    if not threads:
        return None
    return random.choice(threads[:BOARD_WINDOW])["id"]


def act(key, persona, personas, thread, config):
    """One forum turn: look at the board, decide, then perform."""
    model = persona.get("model") or config["default_model"]
    thread_id = None

    # Pick a thread and read it fresh each turn, so a persona sees anything
    # posted since their last look - including a thread a real person
    # started after the simulation was already running.
    if LIVE_FORUM:
        thread_id = pick_thread(LIVE_FORUM)
        if thread_id:
            try:
                thread[:] = forum.read_thread(LIVE_FORUM, thread_id)
            except Exception:
                thread_id = None    # keep the copy we already have

    thread_text = render_thread(thread)

    situation = with_memory(
        persona,
        f"You are looking at a thread on your company's internal forum:\n\n{thread_text}",
    )
    choice = decide(persona, situation, model, config["ollama_url"])
    print(f"[{persona['username']} decides] {choice['action']}"
          f"{' -> ' + choice['target'] if choice['target'] else ''}"
          f"  ({choice['reason']})")

    if choice["action"] == "do_nothing":
        print("   ...says nothing.\n")
        return

    context = (thread_text if choice["action"] == "reply_to_thread"
               else f"You have been reading this thread:\n\n{thread_text}")
    topic = thread[0]["text"][:70] if thread else "whatever they chose to raise"
    perform(key, persona, personas, choice, context,
            topic=topic, config=config, thread=thread, thread_id=thread_id)


def _same_text(a, b):
    """Compare two posts ignoring how their line breaks happen to fall."""
    return " ".join(a.split()) == " ".join(b.split())


def seed_board(forum_url, days):
    """Put each configured opening post on the forum, once.

    Anything already on the board is left alone, so restarting the service
    carries on with the threads that are there instead of pinning up a
    second copy of the same note.
    """
    try:
        existing = [thread["content"] for thread in forum.list_threads(forum_url)]
    except Exception as problem:
        print(f"   (could not read the board: {problem})\n")
        return

    for day in days:
        post = day.get("post")
        if not post:
            continue
        if any(_same_text(post["text"], other) for other in existing):
            print(f"   (already on the board: {day['name']})")
            continue
        try:
            thread_id = forum.start_thread(forum_url, post["username"], post["text"])
            print(f"   (pinned up {day['name']} as thread {thread_id})")
        except Exception as problem:
            print(f"   (could not start a thread: {problem})")


def main():
    global LIVE_FORUM

    config = load_config()
    if not check_connection(config["ollama_url"]):
        print("Cannot reach Ollama. Is it running?")
        return

    if RESET_MEMORY:
        forget_everyone()

    people = load_people()
    generated = len(people) - len(HAND_WRITTEN)
    print(f"{len(people)} personas on the forum "
          f"({len(HAND_WRITTEN)} written by hand, {generated} generated).\n")

    turns_per_day = config.get("turns_per_day", 4)

    # How long to wait between actions, and between visits to the forum.
    min_gap = config.get("min_gap", 30)
    max_gap = config.get("max_gap", 120)
    min_away = config.get("min_away", 600)
    max_away = config.get("max_away", 1800)

    # No days configured means an empty forum, so the personas start their
    # own conversations instead of reacting to a seeded post.
    days = load_days() or [{"name": "Day 1", "post": None}]

    # Where the lab forum is. Empty or unreachable means the thread is kept
    # in memory instead, so this still runs on a laptop.
    forum_url = config.get("forum_url")
    if forum_url and forum.is_up(forum_url):
        LIVE_FORUM = forum_url
        print(f"Using the forum at {forum_url}\n")
        seed_board(forum_url, days)
    else:
        print("No forum reachable - keeping the thread in memory.\n")

    # With a real forum the personas choose what to read each turn, so the
    # configured days only seed the board. Without one, each day opens the
    # next in-memory thread.
    to_start = [] if LIVE_FORUM else list(days)
    thread = []
    visit = 0

    while True:
        visit += 1
        if to_start:
            day = to_start.pop(0)
            print(f"\n==================== {day['name']} ====================")
            thread = []
            if day.get("post"):
                thread.append(day["post"])
                print(f"{day['post']['username']}: {day['post']['text']}\n")
            else:
                print("(nothing on the forum yet)\n")
        else:
            print(f"\n==================== Visit {visit} ====================")

        for _ in range(turns_per_day):
            # A different few people look in each round, so the same voices
            # do not answer everything.
            readers = random.sample(sorted(people),
                                    min(READERS_PER_VISIT, len(people)))
            print(f"(reading now: {', '.join(readers)})\n")

            for key in readers:
                act(key, people[key], people, thread, config)
                pause(min_gap, max_gap, "reading")

        pause(min_away, max_away, "away from the forum")


if __name__ == "__main__":
    main()
