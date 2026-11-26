"""
Builds a folder of persona files from one group description.

Writing fifteen people by hand does not scale, so company/group.yaml says
how many people you want and gives hints for the fields that matter. This
script turns that into real persona files in company/personas/, which you
can open and read like any hand-written one.

How a hint is used:
  a list      - pick one of exactly these, no model involved
  a sentence  - ask the model once for a pool of options, then pick from it
  left out    - ask the model for a pool with no constraints

Asking for a pool once per field, rather than once per person, is what
keeps this fast. Fifteen people cost about three model calls, not fifteen.

Run it:  python make_people.py
"""

import random
import re
import sys
from pathlib import Path

import yaml

from llm import chat_json, check_connection
from persona import load_config

HERE = Path(__file__).parent / "company"
OUT = HERE / "personas"


def ask_for_pool(what, hint, count, model, ollama_url):
    """Ask the model for a list of options for one field.

    Returns a list of strings. The hint goes in verbatim, so "Italian
    names" gives Italian names and nothing else constrains it.
    """
    instruction = (
        f"Give {count} different options for the '{what}' field of a "
        f"fictional person."
    )
    if hint:
        instruction += f" They must match this: {hint}."
    instruction += ' Reply with JSON only: {"options": ["...", "..."]}'

    raw = chat_json(
        "You generate lists of realistic but fictional details. "
        "No real people. Reply with JSON only.",
        instruction,
        model=model,
        ollama_url=ollama_url,
        temperature=0.9,
    )

    options = (raw or {}).get("options") or []
    return [str(o).strip() for o in options if str(o).strip()]


def build_pool(field, hint, size, model, ollama_url):
    """Turn one hint into a pool of options to pick from."""
    if isinstance(hint, list):
        return list(hint)                      # exact options, no model needed

    # Ask for more than we need, because the model repeats itself.
    wanted = max(size * 2, 10)
    pool = ask_for_pool(field, hint, wanted, model, ollama_url)
    if not pool:
        print(f"  the model gave nothing usable for '{field}'")
        sys.exit(1)
    return pool


def make_username(full_name, taken):
    """First initial plus surname, the way a work login usually looks."""
    parts = re.findall(r"[A-Za-z]+", full_name)
    if len(parts) >= 2:
        base = (parts[0][0] + parts[-1]).lower()
    else:
        base = (parts[0] if parts else "user").lower()

    username = base
    n = 2
    while username in taken:                   # two Smiths in one hospital
        username = f"{base}{n}"
        n += 1
    taken.add(username)
    return username


def main():
    group = yaml.safe_load(open(HERE / "group.yaml", encoding="utf-8"))
    config = load_config()
    model = config["default_model"]
    url = config["ollama_url"]

    if not check_connection(url):
        print("Cannot reach Ollama. Is it running?")
        return

    size = group["size"]
    hints = group.get("hints", {}) or {}
    print(f"Building {size} people for {group['name']}...\n")

    # One pool per field, shared by everyone in the group.
    pools = {}
    for field in ("name", "job", "employer"):
        hint = hints.get(field)
        kind = "list" if isinstance(hint, list) else "model"
        print(f"  {field}: asking for options ({kind})")
        pools[field] = build_pool(field, hint, size, model, url)

    names = list(dict.fromkeys(pools["name"]))  # drop repeats, keep order
    if len(names) < size:
        print(f"\nOnly got {len(names)} different names for {size} people. "
              f"Run it again, or narrow the name hint.")
        size = len(names)

    random.shuffle(names)
    sides = group["stance"]["sides"]
    topic = group["stance"]["topic"]

    # Everyone on a side agreeing for the same reason reads like one person
    # with fifteen usernames, so ask for a pool of reasons per side and give
    # each person a different one. Two calls, not fifteen.
    reasons = {}
    for side in sides:
        print(f"  reasons for people {side} it")
        reasons[side] = ask_for_pool(
            "reason",
            f"short, specific reasons a hospital worker would be {side} "
            f"{topic}. One short sentence each, written in first person",
            max(size, 8), model, url,
        ) or [f"{side} it"]

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.yaml"):
        old.unlink()

    taken = set()
    print()
    for i in range(size):
        name = names[i]
        job = random.choice(pools["job"])
        employer = random.choice(pools["employer"])
        side = sides[i % len(sides)]            # split the group evenly
        username = make_username(name, taken)

        person = {
            "name": name,
            "username": username,
            "job": job,
            "employer": employer,
            "background": f"{random.randint(1, 20)} years as a {job.lower()}",
            "traits": [random.choice(group["traits"])],
            "vocabulary": random.choice(group["vocabulary"]),
            "opinions": {
                "position": f"{side} {topic}",
                "reasoning": reasons[side][i % len(reasons[side])],
            },
            "stance": {"topic": topic, "side": side},
            "apps": list(group.get("apps", ["forum"])),
            "model": None,
        }

        path = OUT / f"{username}.yaml"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Generated by make_people.py from company/group.yaml.\n"
                    f"# Edit freely - it will be overwritten next time you run it.\n\n")
            yaml.dump(person, f, sort_keys=False, allow_unicode=True)

        print(f"  {username:14} {name:24} {job:20} {side}")

    print(f"\n{size} people written to company/personas/")


if __name__ == "__main__":
    main()
