"""Synthetic contextual state-machine checks, not native visibility evidence."""
from dataclasses import FrozenInstanceError, replace
import unittest

from experiments.learner.contextual import (
    Action, Context, Frame, Hero, ObjectView, Observation, Offers, PlayerHistory,
    Result, Stack, Tile, Town, Verification, VisibleEvent, Window,
    decode_frame, decode_observation, encode_frame, finalize_cutoff, finalize_natural,
)


def observation(*, movement=1000, resources=(0,) * 7, window=None, enemy=True):
    return Observation(
        size=(20, 20, 1),
        tiles=(Tile((1, 1, 0), 2, 1, False, False), Tile((2, 1, 0), 2, 0, False, False)),
        objects=(ObjectView("seen-enemy", (2, 1, 0), 34, 0, "opponent"),) if enemy else (),
        resources=resources, heroes=(Hero("own-hero", (1, 1, 0), movement),),
        towns=(Town("own-town", (1, 1, 0)),), window=window,
    )


def hero_window(count=25):
    return Window(Context("hero", "own-hero"), (Stack(0, 0, count),),
                  primary_skills=(1, 2, 3, 4), mana=10, level=1)


def decision(history, obs=None):
    obs = obs or observation()
    return history.emit("decision", observation=obs, offers=Offers.from_observation(obs),
                        revision=history.revision + 1)


def submit(history, action, *, consume=True):
    seq = history.decision.seq
    history.emit("submitted", action=action, decision_seq=seq)
    if consume:
        history.emit("consumed", consumer="adapter", decision_seq=seq)
    return seq


def complete_move(history, target=(1, 1), status="reached", accepted=True):
    start = decision(history)
    seq = submit(history, Action("move_to", "own-hero", *target))
    history.emit("result", decision_seq=seq, result=Result(status, accepted, (1, 1, 0), 1000, target == (1, 1)))
    return start


def terminal(history, outcome):
    if history.decision is not None:
        history.emit("superseded", decision_seq=history.decision.seq, reason="terminal")
    return history.emit("terminal", outcome=outcome)


def finalize(pair):
    return finalize_natural(pair, engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                            process_exit_code=0, evidence_ref="native-evidence.json")


def finalize_own_cutoff(history):
    other = PlayerHistory(history.episode, 1 - history.player)
    held = decision(other)
    other.emit("cutoff", decision_seq=held.seq, reason="budget")
    pair = (history, other) if history.player == 0 else (other, history)
    return finalize_cutoff(pair, evidence_ref="cutoff-evidence", histories_complete=(True, True),
                           failure_free=True, shutdown_verified=True, continuation_limit=10,
                           continuation_steps=(0, 0))


class ContextualContractTests(unittest.TestCase):
    def test_full_coordinate_support_independent_of_fog_points_and_obstacles(self):
        empty = replace(observation(movement=0, enemy=False), tiles=())
        # Own-object map/list access remains valid independently of current terrain.
        offers = Offers.from_observation(empty)
        for y in range(20):
            for x in range(20):
                self.assertTrue(offers.permits(Action("move_to", "own-hero", x, y)))
        self.assertFalse(offers.permits(Action("move_to", "own-hero", 20, 1)))
        self.assertFalse(offers.permits(Action("move_to", "opponent", 1, 1)))
        self.assertEqual(Offers.from_observation(observation()), offers)

    def test_no_precomputed_reachability_or_cost_fields(self):
        history = PlayerHistory("game", 0)
        frame = decision(history)
        for field in ("path_cost", "reachable", "arrival_turn", "route"):
            wire = encode_frame(frame)
            wire["observation"][field] = 100
            with self.assertRaises(ValueError):
                decode_frame(wire)
        wire = encode_frame(frame)
        wire["offers"]["target_mask"] = [[1, 1]]
        with self.assertRaises(ValueError):
            decode_frame(wire)

    def test_nested_raw_ids_unknown_fields_and_other_schema_rejected(self):
        history = PlayerHistory("game", 0)
        frame = decision(history)
        wire = encode_frame(frame)
        wire["observation"]["objects"][0]["engine_id"] = 99
        with self.assertRaises(ValueError):
            decode_frame(wire)
        wire = encode_frame(frame)
        wire["schema"] = "heroes-player-history-v1"
        with self.assertRaises(ValueError):
            decode_frame(wire)
        with self.assertRaises(ValueError):
            decode_frame({"type": "observation", "protocol_version": 4})

    def test_map_cannot_include_army_and_window_must_match_owned_context(self):
        history = PlayerHistory("game", 0)
        frame = decision(history)
        wire = encode_frame(frame)
        wire["observation"]["heroes"][0]["army"] = [{"slot": 0, "creature": 0, "count": 25}]
        with self.assertRaises(ValueError):
            decode_frame(wire)
        with self.assertRaises(ValueError):
            replace(observation(), window=replace(hero_window(), context=Context("hero", "seen-enemy")))
        with self.assertRaises(ValueError):
            replace(observation(), objects=(ObjectView("hidden", (19, 19, 0), 34, 0, "opponent"),))

    def test_window_navigation_close_and_non_refresh_preserve_old_prefix(self):
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("open_hero", "own-hero"))
        history.emit("context", decision_seq=seq, context=Context("hero", "own-hero"), revision=history.revision + 1)
        panel = history.emit("view", observation=observation(window=hero_window()))
        history.emit("result", decision_seq=seq, result=Result("opened", True))
        decision(history, observation(window=hero_window()))
        prior_prefix = history.frames
        seq = submit(history, Action("close"))
        history.emit("context", decision_seq=seq, context=Context(), revision=history.revision + 1)
        history.emit("view", observation=observation(resources=(100,) * 7))
        history.emit("result", decision_seq=seq, result=Result("closed", True))
        self.assertEqual(history.frames[:len(prior_prefix)], prior_prefix)
        self.assertIsNone(history.frames[-2].observation.window)
        self.assertEqual(panel.observation.window.army[0].count, 25)
        self.assertEqual(history.frames[-2].observation.resources, (100,) * 7)
        with self.assertRaises(ValueError):
            history.emit("view", observation=observation(window=hero_window(40)))
        with self.assertRaises(FrozenInstanceError):
            panel.observation.window.army[0].count = 40
        decision(history)
        seq = submit(history, Action("open_hero", "own-hero"))
        history.emit("context", decision_seq=seq, context=Context("hero", "own-hero"), revision=history.revision + 1)
        updated = history.emit("view", observation=observation(window=hero_window(40)))
        self.assertEqual(updated.observation.window.army[0].count, 40)
        self.assertEqual(panel.observation.window.army[0].count, 25)

    def test_town_window_minimal_and_extra_operations_deferred(self):
        town = Window(Context("town", "own-town"), (), faction=0)
        self.assertEqual(Offers.from_observation(observation(window=town)).hero_refs, ())
        self.assertTrue(Offers.from_observation(observation(window=town)).permits(Action("close")))
        with self.assertRaises(ValueError):
            replace(town, buildings=(0, 1))
        for kind in ("construct", "recruit", "transfer_stack", "open_obelisk"):
            with self.assertRaises(ValueError):
                Action(kind)

    def test_native_issuance_is_neither_acknowledgement_nor_completed_sample(self):
        history = PlayerHistory("game", 0)
        first = decision(history)
        seq = submit(history, Action("move_to", "own-hero", 10, 10))
        history.emit("issued", decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=seq, result=Result("movement_limit", True, (1, 1, 0), 0, False))
        self.assertEqual(history.transitions(), ())
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("view", observation=observation(movement=100))
        history.emit("issued", decision_seq=seq, command_index=1)
        history.emit("acknowledged", decision_seq=seq, command_index=1, accepted=False)
        history.emit("result", decision_seq=seq, result=Result("interrupted", True, (1, 1, 0), 100, False))
        next_decision = decision(history, observation(movement=100))
        history.emit("cutoff", decision_seq=next_decision.seq, reason="decision_budget")
        finalize_own_cutoff(history)
        transition, = history.transitions()
        self.assertEqual((transition.decision_seq, transition.successor_seq), (first.seq, next_decision.seq))
        self.assertEqual(transition.reward, 0)
        self.assertTrue(transition.eligible and transition.bootstrap_allowed)

    def test_consumption_and_context_result_order_enforced(self):
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("open_hero", "own-hero"), consume=False)
        with self.assertRaises(ValueError):
            history.emit("context", decision_seq=seq, context=Context("hero", "own-hero"), revision=1)
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=seq, result=Result("opened", True))
        history.emit("consumed", decision_seq=seq, consumer="adapter")
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=seq, result=Result("opened", True))
        with self.assertRaises(ValueError):
            history.emit("issued", decision_seq=seq, command_index=0)
        with self.assertRaises(ValueError):
            history.emit("context", decision_seq=seq, context=Context("town", "own-town"), revision=1)

    def test_superseded_decision_then_final_views_no_prefix_mutation(self):
        pair = (PlayerHistory("game", 0), PlayerHistory("game", 1))
        for history in pair:
            complete_move(history)
            held = decision(history)
            prefix = history.frames
            history.emit("superseded", decision_seq=held.seq, reason="terminal")
            history.emit("view", observation=observation(enemy=False, resources=(5,) * 7))
            self.assertEqual(history.frames[:len(prefix)], prefix)
        terminal(pair[0], "win")
        terminal(pair[1], "loss")
        pair[0].emit("view", observation=observation(enemy=False, resources=(6,) * 7))
        for history in pair:
            self.assertFalse(history.transitions()[0].eligible)
            self.assertIsNone(history.transitions()[0].reward)
        finalize(pair)
        for history, reward in zip(pair, (1, -1)):
            transition, = history.transitions()
            self.assertTrue(transition.eligible)
            self.assertEqual(transition.reward, reward)
            self.assertIsNone(transition.successor_seq)
            self.assertFalse(transition.bootstrap_allowed)
            with self.assertRaises(ValueError):
                history.emit("view", observation=observation())

    def test_unconsumed_and_incompletely_acknowledged_actions_not_samples(self):
        for consume in (False, True):
            pair = (PlayerHistory("game", 0), PlayerHistory("game", 1))
            for history in pair:
                complete_move(history)
                decision(history)
                seq = submit(history, Action("move_to", "own-hero", 10, 10), consume=consume)
                if consume:
                    history.emit("issued", decision_seq=seq, command_index=0)
                history.emit("superseded", decision_seq=seq, reason="terminal")
                history.emit("view", observation=observation(enemy=False))
            terminal(pair[0], "win")
            terminal(pair[1], "loss")
            finalize(pair)
            for history in pair:
                self.assertEqual(len(history.transitions()), 1)
                transition = history.transitions()[0]
                self.assertEqual(transition.decision_seq, 0)
                self.assertEqual(transition.reward, 0 if consume else (1 if history.player == 0 else -1))
                self.assertEqual(transition.bootstrap_allowed, consume)
                self.assertEqual(transition.successor_seq is not None, consume)

    def test_cutoff_requires_real_own_unacted_decision(self):
        history = PlayerHistory("game", 0)
        with self.assertRaises(ValueError):
            history.emit("cutoff", decision_seq=0, reason="budget")
        complete_move(history)
        next_decision = decision(history)
        seq = submit(history, Action("end_turn"), consume=False)
        with self.assertRaises(ValueError):
            history.emit("cutoff", decision_seq=next_decision.seq, reason="budget")
        history.emit("superseded", decision_seq=seq, reason="stale")
        with self.assertRaises(ValueError):
            history.emit("cutoff", decision_seq=next_decision.seq, reason="budget")

    def test_failure_after_terminal_cannot_be_reclassified_natural(self):
        pair = (PlayerHistory("game", 0), PlayerHistory("game", 1))
        for history in pair:
            complete_move(history)
        terminal(pair[0], "win")
        terminal(pair[1], "loss")
        pair[0].emit("failure", reason="reader_error")
        with self.assertRaises(ValueError):
            finalize(pair)
        transition, = pair[0].transitions()
        self.assertFalse(transition.eligible or transition.bootstrap_allowed)
        self.assertIsNone(transition.reward)

    def test_terminal_does_not_silently_drop_pending_action_or_held_decision(self):
        history = PlayerHistory("game", 0)
        decision(history)
        with self.assertRaises(ValueError):
            history.emit("terminal", outcome="win")
        seq = submit(history, Action("end_turn"))
        with self.assertRaises(ValueError):
            history.emit("terminal", outcome="win")
        history.emit("superseded", decision_seq=seq, reason="terminal")
        history.emit("terminal", outcome="win")
        with self.assertRaises(ValueError):
            decision(history)

    def test_cross_player_cross_episode_and_sequence_leaks_rejected(self):
        first = PlayerHistory("game", 0)
        frame = decision(first)
        other = PlayerHistory("game", 1)
        other_episode = PlayerHistory("other", 0)
        for history in (other, other_episode, first):
            with self.assertRaises(ValueError):
                history.append(frame)
        first.emit("superseded", decision_seq=frame.seq, reason="stale")
        with self.assertRaises(ValueError):
            first.append(replace(frame, seq=20, revision=20))
        with self.assertRaises(ValueError):
            first.emit("view", day=0, observation=observation())

    def test_frozen_wire_roundtrip_copies_and_empty_payloads_fail_closed(self):
        history = PlayerHistory("game", 0)
        frame = decision(history)
        wire = encode_frame(frame)
        self.assertEqual(decode_frame(wire), frame)
        wire["observation"]["heroes"][0]["movement"] = 0
        self.assertEqual(frame.observation.heroes[0].movement, 1000)
        for raw in ({}, {"size": [20, 20, 1]}, {"window": {}}):
            with self.assertRaises(ValueError):
                decode_observation(raw)
        with self.assertRaises(ValueError):
            replace(frame, reason="hidden data")

    def test_transient_enemy_sighting_preserved_between_decisions(self):
        history = PlayerHistory("game", 0)
        decision(history, observation(enemy=False))
        seq = submit(history, Action("end_turn"))
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("view", observation=observation(enemy=True))
        history.emit("event", event=VisibleEvent("battle_start", side="attacker"))
        history.emit("view", observation=observation(enemy=False))
        history.emit("event", event=VisibleEvent("battle_complete", won=True))
        history.emit("result", decision_seq=seq, result=Result("turn_ended", True))
        last = decision(history, observation(enemy=False))
        history.emit("cutoff", decision_seq=last.seq, reason="budget")
        self.assertEqual(sum(bool(f.observation and f.observation.objects) for f in history.frames), 1)
        self.assertEqual([f.event.kind for f in history.frames if f.event], ["battle_start", "battle_complete"])

    def test_finalizer_rejects_mismatch_crash_incomplete_or_wrong_digest(self):
        pair = (PlayerHistory("game", 0), PlayerHistory("game", 1))
        terminal(pair[0], "win")
        terminal(pair[1], "loss")
        common = dict(engine_outcomes=("win", "loss"), socket_outcomes=("win", "loss"),
                      process_exit_code=0, evidence_ref="evidence")
        for overrides in ({"process_exit_code": -9}, {"engine_outcomes": ("win", "win")},
                          {"histories_complete": (True, False)}, {"socket_outcomes": ("loss", "win")}):
            with self.assertRaises(ValueError):
                finalize_natural(pair, **(common | overrides))
            self.assertIsNone(pair[0].boundary)
        proof = Verification("game", "evidence", ("win", "loss"), ("win", "loss"), 0,
                             (True, True), ("0" * 64, "0" * 64))
        with self.assertRaises(ValueError):
            pair[0].emit("finalized", verification=proof)
        self.assertEqual(pair[0].frames[-1].kind, "terminal")

    def test_observable_movement_result_must_match_target_and_bounds(self):
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("move_to", "own-hero", 3, 3))
        for result in (Result("reached", True, (1, 1, 0), 900, True),
                       Result("movement_limit", True),
                       Result("movement_limit", True, (25, 1, 0), 0, False)):
            with self.assertRaises(ValueError):
                history.emit("result", decision_seq=seq, result=result)
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=Result("movement_limit", True, (2, 1, 0), 50, False))

    def test_next_transition_is_same_player_genuine_next_decision(self):
        history = PlayerHistory("game", 1)
        first = complete_move(history)
        stale = decision(history)
        history.emit("superseded", decision_seq=stale.seq, reason="visible_change")
        history.emit("view", observation=observation(enemy=False))
        second = complete_move(history)
        cutoff = decision(history)
        history.emit("cutoff", decision_seq=cutoff.seq, reason="budget")
        finalize_own_cutoff(history)
        transitions = history.transitions()
        self.assertEqual([(t.player, t.decision_seq, t.successor_seq) for t in transitions],
                         [(1, first.seq, second.seq), (1, second.seq, cutoff.seq)])
        self.assertTrue(all(t.reward == 0 and t.bootstrap_allowed for t in transitions))


    def test_explicit_window_invalidation_after_supersession_and_terminal(self):
        for after_terminal in (False, True):
            history = PlayerHistory("game", 0)
            decision(history)
            seq = submit(history, Action("open_hero", "own-hero"))
            history.emit("context", decision_seq=seq, context=Context("hero", "own-hero"), revision=1)
            history.emit("view", observation=observation(window=hero_window()))
            history.emit("result", decision_seq=seq, result=Result("opened", True))
            held = decision(history, observation(window=hero_window()))
            with self.assertRaises(ValueError):
                history.emit("context_invalidated", context=Context(), revision=3, reason="subject_unavailable")
            history.emit("superseded", decision_seq=held.seq, reason="terminal")
            if after_terminal:
                history.emit("terminal", outcome="loss")
            history.emit("context_invalidated", context=Context(), revision=3,
                         reason="terminal" if after_terminal else "subject_unavailable")
            history.emit("view", observation=replace(observation(enemy=False), heroes=()))
            self.assertEqual(history.context, Context())
            self.assertIsNone(history.frames[-1].observation.window)
            self.assertEqual(history.frames[4].observation.window.army[0].count, 25)

    def test_result_status_and_acceptance_consistency(self):
        for status, accepted in (("rejected", True), ("opened", False), ("closed", False),
                                 ("turn_ended", False), ("reached", False)):
            with self.assertRaises(ValueError):
                Result(status, accepted)


    def test_turn_completion_requires_exactly_one_accepted_command(self):
        for acknowledged in (None, False):
            history = PlayerHistory("game", 0)
            decision(history)
            seq = submit(history, Action("end_turn"))
            if acknowledged is not None:
                history.emit("issued", decision_seq=seq, command_index=0)
                history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=acknowledged)
            with self.assertRaises(ValueError):
                history.emit("result", decision_seq=seq, result=Result("turn_ended", True))
            history.emit("result", decision_seq=seq, result=Result("rejected", False))
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("end_turn"))
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=True)
        history.emit("result", decision_seq=seq, result=Result("turn_ended", True))

    def test_movement_cannot_claim_effect_without_native_acceptance(self):
        for acknowledged in (None, False):
            history = PlayerHistory("game", 0)
            decision(history)
            seq = submit(history, Action("move_to", "own-hero", 2, 1))
            if acknowledged is not None:
                history.emit("issued", decision_seq=seq, command_index=0)
                history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=acknowledged)
            for result in (Result("reached", True, (2, 1, 0), 900, True),
                           Result("interrupted", True, (1, 1, 0), 900, False),
                           Result("interaction", True, (1, 1, 0), 1000, False)):
                with self.assertRaises(ValueError):
                    history.emit("result", decision_seq=seq, result=result)
            history.emit("result", decision_seq=seq, result=Result("interrupted", True, (1, 1, 0), 1000, False))

    def test_rejected_navigation_cannot_follow_context_change(self):
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("open_hero", "own-hero"))
        history.emit("context", decision_seq=seq, context=Context("hero", "own-hero"), revision=1)
        with self.assertRaises(ValueError):
            history.emit("result", decision_seq=seq, result=Result("rejected", False))
        history.emit("result", decision_seq=seq, result=Result("opened", True))


    def test_cutoff_is_provisional_until_both_players_finish_without_failure(self):
        red, blue = PlayerHistory("game", 0), PlayerHistory("game", 1)
        complete_move(red)
        held = decision(red)
        prefix = red.frames
        red.emit("cutoff", decision_seq=held.seq, reason="budget")
        self.assertEqual(red.boundary, "cutoff_pending")
        self.assertFalse(red.transitions()[0].eligible)
        self.assertIsNone(red.transitions()[0].reward)
        blue.emit("failure", reason="reader_error")
        with self.assertRaises(ValueError):
            finalize_cutoff((red, blue), evidence_ref="bad", histories_complete=(True, False),
                            failure_free=False, shutdown_verified=True, continuation_limit=1,
                            continuation_steps=(0, 0))
        red.emit("failure", reason="peer_reader_error")
        self.assertEqual(red.frames[:len(prefix)], prefix)
        self.assertFalse(red.transitions()[0].eligible or red.transitions()[0].bootstrap_allowed)

    def test_pair_cutoff_proof_roundtrip_and_declared_continuation_bound(self):
        red, blue = PlayerHistory("game", 0), PlayerHistory("game", 1)
        for history in (red, blue):
            complete_move(history)
            held = decision(history)
            history.emit("cutoff", decision_seq=held.seq, reason="budget")
        common = dict(evidence_ref="cutoff", histories_complete=(True, True), failure_free=True,
                      shutdown_verified=True, continuation_limit=2, continuation_steps=(1, 0))
        for overrides in ({"failure_free": False}, {"shutdown_verified": False},
                          {"histories_complete": (True, False)}, {"continuation_steps": (3, 0)}):
            with self.assertRaises(ValueError):
                finalize_cutoff((red, blue), **(common | overrides))
            self.assertFalse(red.transitions()[0].eligible)
        finalize_cutoff((red, blue), **common)
        for history in (red, blue):
            self.assertTrue(history.transitions()[0].eligible)
            self.assertEqual(history.transitions()[0].reward, 0)
            self.assertTrue(history.transitions()[0].bootstrap_allowed)
            replayed = PlayerHistory("game", history.player)
            for frame in history.frames:
                replayed.append(decode_frame(encode_frame(frame)))
            self.assertEqual(replayed.transitions(), history.transitions())

    def test_missing_explicit_wire_schema_rejected(self):
        history = PlayerHistory("game", 0)
        frame = decision(history)
        for field in ("schema", "episode", "player", "seq", "day", "kind"):
            wire = encode_frame(frame)
            del wire[field]
            with self.assertRaises(ValueError):
                decode_frame(wire)

    def test_rejected_native_step_forbids_automatic_next_step(self):
        history = PlayerHistory("game", 0)
        decision(history)
        seq = submit(history, Action("move_to", "own-hero", 10, 10))
        history.emit("issued", decision_seq=seq, command_index=0)
        history.emit("acknowledged", decision_seq=seq, command_index=0, accepted=False)
        with self.assertRaises(ValueError):
            history.emit("issued", decision_seq=seq, command_index=1)
        history.emit("result", decision_seq=seq, result=Result("interrupted", True, (1, 1, 0), 1000, False))


if __name__ == "__main__":
    unittest.main()
