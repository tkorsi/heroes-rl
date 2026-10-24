"""One player's stream in the learner's ``heroes-player-history-v1`` format.

The learner contract (``experiments/learner/contract.py``, issue #4) defines
four frame kinds: a complete current ``view`` at each observable change, a
``decision`` with its view and nine-entry attempt mask, the player's own
``action`` and its ``result``. This module emits exactly those JSON records
from the sanitized history that ``PlayerHistory`` folds. It adds no field of
its own: the sequence counts only this player's frames, the day is the public
game day and object identities are the player's opaque handles.

Visible information that v1 cannot represent (battle start/result details,
visits, dialogs and their uniform answers) stays in the player's full record
stream and is counted here, never silently dropped. A frame order that v1
forbids, such as a visible change after an unacted decision, makes this
projection invalid for the episode; the projection then stops and says why,
and the simulation result itself is unaffected.
"""
from __future__ import annotations

SCHEMA = "heroes-player-history-v1"
PLAYER_COLORS = 8  # VCMI player colors 0..7; -1 is neutral and -2 unflaggable (PlayerColor in the pinned VCMI)


def relative_owner(owner: int, own_color: int) -> str:
    if owner == own_color:
        return "self"
    return "opponent" if 0 <= owner < PLAYER_COLORS else "neutral"


class HistoryV1:
    def __init__(self, episode: str, player: int):
        self.episode, self.player = episode, player
        self.frames: list[dict] = []
        self.decision: int | None = None  # sequence of an emitted, unacted decision
        self.pending: tuple[int, int] | None = None  # (decision sequence, action) awaiting its result
        self.day = 1
        self.error: str | None = None
        self.unrepresented: dict[str, int] = {}

    def view(self, view: dict, day: int) -> None:
        if self.decision is not None:
            self._fail("A visible change arrived after an unacted decision")
        self._emit("view", day, view=view)

    def decision_frame(self, view: dict, mask: list[bool], day: int) -> None:
        if self.decision is not None or self.pending is not None:
            self._fail("A decision arrived before the previous action and result completed")
        if self._emit("decision", day, view=view, mask=list(mask)):
            self.decision = self.frames[-1]["seq"]

    def action(self, action: int, day: int) -> None:
        if self.decision is None:
            self._fail("An action has no emitted decision")
        decision = self.decision
        if self._emit("action", day, action=action, decision_seq=decision):
            self.decision, self.pending = None, (decision, action)

    def result(self, action: int, accepted: bool, day: int) -> None:
        if self.pending is None or self.pending[1] != action:
            self._fail("A result does not match the pending action")
        decision = self.pending[0] if self.pending else None
        if self._emit("result", day, action=action, accepted=accepted, decision_seq=decision):
            self.pending = None

    def note_unrepresented(self, kind: str) -> None:
        self.unrepresented[kind] = self.unrepresented.get(kind, 0) + 1

    def summary(self) -> dict:
        return {"schema": SCHEMA, "frames": len(self.frames), "last_seq": len(self.frames) - 1 if self.frames else None,
                "valid": self.error is None, "error": self.error, "unacted_decision": self.decision,
                "pending_action": list(self.pending) if self.pending else None,
                "unrepresented_events": dict(sorted(self.unrepresented.items()))}

    def _fail(self, reason: str) -> None:
        if self.error is None:
            self.error = f"{reason} at player sequence {len(self.frames)}"

    def _emit(self, kind: str, day: int, **payload) -> bool:
        if self.error is None and (type(day) is not int or day < self.day):
            self._fail("The public game day is missing or decreased")
        if self.error is not None:
            return False
        self.frames.append({"schema": SCHEMA, "episode": self.episode, "player": self.player,
                            "seq": len(self.frames), "day": day, "kind": kind, **payload})
        self.day = day
        return True
