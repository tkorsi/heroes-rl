"""Fixed baseline agents for the headless environment. They do not learn."""
from __future__ import annotations

import random

ENGINE_ID_KEYS = {"id", "object_id", "hero_id"}


def engine_id_paths(value, path: str = "") -> list[str]:
    """Paths of any raw engine-ID field; a learner record must have none."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ENGINE_ID_KEYS:
                found.append(f"{path}.{key}")
            found.extend(engine_id_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(engine_id_paths(item, f"{path}[{index}]"))
    return found


class UniformRandomAgent:
    """Uniform over permitted attempts and offered dialog choices; reads nothing else."""

    def __init__(self, seed: int):
        if type(seed) is not int:
            raise ValueError("Use an integer agent seed")
        self.seed = seed
        self.rng = random.Random(seed)

    @property
    def description(self) -> dict:
        return {"label": "Uniform random agent (no learning)", "algorithm": "uniform_random", "seed": self.seed,
                "learning": False, "checkpoint": None,
                "action_space": "8 adjacent move attempts + end turn; uniform over permitted attempts",
                "dialog_policy": "uniform over offered choices"}

    def act(self, request) -> int:
        if request.kind == "action":
            return self.rng.choice([index for index, allowed in enumerate(request.mask) if allowed])
        return self.rng.choice(request.choices)
