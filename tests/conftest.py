"""Shared fixtures: a scripted stand-in for the engine, so tests need no VCMI."""
import json
from pathlib import Path
import stat
import sys

import pytest

from heroes.engine.process import Engine

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_engine(tmp_path, monkeypatch):
    """An ``Engine`` whose client is ``tests/fake_vcmi.py``; call it with a script to choose the game."""
    binary = tmp_path / "runtime" / "vcmiclient"
    binary.parent.mkdir()
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).with_name("fake_vcmi.py")}" "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "data" / "Data").mkdir(parents=True)

    def make(**script):
        monkeypatch.setenv("FAKE_VCMI", json.dumps(script))
        return Engine(binary, tmp_path / "data", REPOSITORY / "maps" / "hero-management-v4.vmap")

    return make


@pytest.fixture
def play_fake(fake_engine, tmp_path):
    """Play one scripted game with a fresh random policy; returns the finished ``Game``."""
    from heroes import actor, checkpoint
    from heroes.engine.env import Env, Limits
    from heroes.game.guard import STACK_CYCLE_VERSION
    model = checkpoint.fresh(learning_rate=3e-4).model.eval()
    count = iter(range(1000))

    def run(learner=0, limits=Limits(), **script):
        env = Env(fake_engine(**script), limits)
        try:
            number = next(count) + 1
            return actor.play(env, model, actor.Job(number, 7, learner, (1, 2)), tmp_path / f"game-{number}",
                              guard=STACK_CYCLE_VERSION)
        finally:
            env.close()

    run.model = model
    return run
