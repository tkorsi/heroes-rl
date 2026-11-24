"""Start positions for practice games: which map file a game begins from.

An ordinary game begins on the training map as it is. A run told ``--starts
FILE`` begins every game from one of the positions the file lists instead:
the training map with the two heroes elsewhere (``tools/make_starts.py``).
The file is a mix, written by whoever decides what is practised now:

    {"schema": "heroes-start-mix-v1",
     "starts": [{"name": "level1-red-below", "file": "/…/level1-red-below.vmap", "weight": 1.0}, …]}

A game's position is drawn with the weights given, from the run's seeded
generator when it has one. Nothing else about a game changes: the engine, the
rules, the rewards and what the policy may do are those of every other game.

Two things an entry may also say. ``"file": null`` is the run's own map: the
ordinary start, as one of the starts of the mix. ``"max_days": N`` stops the
games from that entry after day N, whatever the run's limit is; without it a
game runs under the run's limit. A mix can so hold practice positions that
are stopped early beside ordinary games that are not.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

SCHEMA = "heroes-start-mix-v1"


@dataclass(frozen=True)
class Starts:
    names: tuple  # of the positions
    files: tuple  # their map files; None for the run's own map
    weights: tuple  # how often each is drawn, relative to the others
    days: tuple = ()  # the day after which a game from each is stopped; None for the run's limit

    def choose(self, generator=None):
        """``(name, file, day limit)`` of the position the next game begins from."""
        generator = generator or _SYSTEM
        index = generator.choices(range(len(self.names)), weights=self.weights)[0]
        return self.names[index], self.files[index], self.days[index] if self.days else None

    def shares(self):
        total = sum(self.weights)
        return {name: round(weight / total, 4) for name, weight in zip(self.names, self.weights)}


_SYSTEM = random.SystemRandom()


def load(path):
    """The mix in the file ``path``; a missing map file or a weight that is not positive is refused."""
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or not data.get("starts"):
        raise ValueError(f"{path} is not a list of start positions ({SCHEMA})")
    names, files, weights, days = [], [], [], []
    for entry in data["starts"]:
        file = entry["file"]
        if file is not None:  # None: the run's own map
            file = Path(file).expanduser()
            file = file if file.is_absolute() else path.parent / file
            if not file.is_file():
                raise FileNotFoundError(f"Start position not found: {file}")
            file = str(file.resolve())
        weight = float(entry.get("weight", 1.0))
        if not weight > 0:
            raise ValueError(f"The weight of {entry['name']} must be positive")
        limit = entry.get("max_days")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            raise ValueError(f"The day limit of {entry['name']} must be a whole number of days, 1 or more")
        names.append(str(entry["name"]))
        files.append(file)
        weights.append(weight)
        days.append(limit)
    if len(set(names)) != len(names):
        raise ValueError(f"{path} names a start position twice")
    return Starts(tuple(names), tuple(files), tuple(weights), tuple(days))


def mix(manifest, levels, out):
    """Write the mix file ``out``: the positions of ``manifest`` whose level is in ``levels`` (level -> its share).

    The positions of one level share that level's share equally. Returns the path written.
    """
    manifest = Path(manifest).expanduser().resolve()
    listed = json.loads(manifest.read_text())["starts"]
    entries = []
    for level, share in sorted(levels.items()):
        chosen = [entry for entry in listed if entry["level"] == int(level)]
        if not chosen:
            raise ValueError(f"{manifest} has no position of level {level}")
        entries.extend({"name": entry["name"], "file": str(manifest.parent / entry["file"]),
                        "weight": float(share) / len(chosen)} for entry in chosen)
    out = Path(out).expanduser().resolve()
    out.write_text(json.dumps({"schema": SCHEMA, "manifest": str(manifest),
                               "levels": {str(level): share for level, share in sorted(levels.items())},
                               "starts": entries}, indent=1) + "\n")
    return out


ORDINARY = "ordinary"  # the name of the run's own map among the starts of a mix


def compose(parts, out):
    """Write the mix file ``out`` from ``parts`` and return its path.

    A part is ``{"share": s, "manifest": path, "level": n, "max_days": d}``: the
    positions of that level in that manifest, sharing ``s`` equally and stopped
    after day ``d`` (None: the run's limit); or ``{"share": s, "ordinary": True}``:
    the run's own map under the run's limit. The shares are relative to each
    other. A position's name is its manifest's folder and its own name, so
    that sets with the same names can be mixed.
    """
    entries, said = [], []
    for part in parts:
        share = float(part["share"])
        if not share > 0:
            continue
        if part.get("ordinary"):
            entries.append({"name": ORDINARY, "file": None, "weight": share, "max_days": None})
            said.append({"ordinary": True, "share": share})
            continue
        manifest = Path(part["manifest"]).expanduser().resolve()
        listed = json.loads(manifest.read_text())
        if listed.get("held_out"):
            raise ValueError(f"{manifest} lists layouts that training never sees: it cannot be part of a mix")
        chosen = [entry for entry in listed["starts"] if entry["level"] == int(part["level"])]
        if not chosen:
            raise ValueError(f"{manifest} has no position of level {part['level']}")
        entries.extend({"name": f"{manifest.parent.name}/{entry['name']}", "file": str(manifest.parent / entry["file"]),
                        "weight": share / len(chosen), "max_days": part.get("max_days")} for entry in chosen)
        said.append({"manifest": str(manifest), "level": int(part["level"]), "share": share, "max_days": part.get("max_days")})
    if not entries:
        raise ValueError("A mix needs a part with a positive share")
    out = Path(out).expanduser().resolve()
    out.write_text(json.dumps({"schema": SCHEMA, "parts": said, "starts": entries}, indent=1) + "\n")
    return out
