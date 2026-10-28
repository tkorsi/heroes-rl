"""Fresh, plan-bound single-learner admission for an explicitly authorized pilot.

Historical diagnostic artifacts never enter this gate. Native opponent commands
remain evidence; there is no opponent policy, value estimate or trajectory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import secrets

import torch

from experiments.headless import builtin_evidence as evidence
from experiments.headless import contextual_evidence as native
from . import contextual as c
from .contextual_model import weights_sha256
from .ppo import PPOConfig, _update
from .builtin_rollout import (BuiltinCollectedEpisode, BuiltinCollector, load_builtin_episode,
                             _opponent_token)
from .contextual_rollout import _frame_bytes

PROVENANCE = "approved_builtin_gameplay_v1"
BINDING_SCHEMA = "heroes-builtin-training-binding-v1"
CAPTURE_SCHEMA = "heroes-builtin-training-capture-v1"
RECEIPT_SCHEMA = "heroes-builtin-training-receipt-v1"
ROLLOUT_SCHEMA = "heroes-builtin-training-rollout-v1"
# These are process-local capabilities. Files document them but cannot recreate
# them. In particular an old diagnostic plus edited labels cannot mint a receipt.
_BINDINGS, _CAPTURES, _RECEIPTS = {}, {}, {}
_USED_RECEIPTS = set()
_PARALLEL_CAPTURES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    return native.file_identity(Path(path))["sha256"]


def _time_left(remaining):
    value = remaining()
    require(type(value) in (int, float), "Expected remaining wall seconds")
    if not 0 < value < float("inf"):
        raise TimeoutError("Shared pilot deadline reached")
    return value


@dataclass(frozen=True)
class TrainingBinding:
    plan_path: str
    plan_sha256: str
    schedule_index: int
    row_json: str
    folder: str
    checkpoint: str
    checkpoint_file_json: str
    weights_sha256: str
    opponent_json: str
    source_json: str
    native_identity_json: str
    max_days: int
    nominal_choices: int = 248
    max_choices: int = 256

    @classmethod
    def create(cls, *, plan_path, plan_sha256, schedule_index, folder, checkpoint, checkpoint_file,
               weights_sha256, opponent_identity, source_identity, native_identity, max_days,
               nominal_choices=248, max_choices=256):
        plan_path, folder, checkpoint_file = (Path(p).resolve() for p in (plan_path, folder, checkpoint_file))
        require(not folder.exists(), "Fresh training collection requires a previously absent episode directory")
        require(digest(plan_path) == plan_sha256, "Training plan bytes differ from the approved plan")
        plan = native.strict_json(plan_path.read_bytes())
        require(plan.get("schema") in ("heroes-builtin-ppo-pilot-v1-plan", "heroes-builtin-training-v1-plan")
                and type(schedule_index) is int and 0 <= schedule_index < len(plan["schedule"]), "Invalid training plan/slot")
        row = plan["schedule"][schedule_index]
        result = cls(str(plan_path), plan_sha256, schedule_index, canonical(row), str(folder), checkpoint,
            canonical({"path": str(checkpoint_file), **native.file_identity(checkpoint_file)}), weights_sha256,
            canonical(opponent_identity), canonical(source_identity), canonical(native_identity), max_days,
            nominal_choices, max_choices)
        result.validate()
        _BINDINGS[id(result)] = result
        return result

    @property
    def row(self): return json.loads(self.row_json)

    @property
    def opponent_identity(self): return json.loads(self.opponent_json)

    def validate(self):
        from .contextual_rollout import _hash, _text
        for value in (self.plan_sha256, self.weights_sha256): _hash(value)
        _text(self.checkpoint, "checkpoint")
        c._integer(self.max_days, low=1)
        require(type(self.nominal_choices) is int and (self.nominal_choices, self.max_choices) in ((248, 256), (4960, 5120))
                and type(self.max_choices) is int,
                "Use the original pilot or explicit 20x continuation limits")
        row = self.row
        require(type(row) is dict and row.get("phase") == "training", "Only a planned training slot can collect training data")
        for name, low, high in (("episode", 1, 2**31-1), ("engine_seed", 1, 2**31-1),
                                ("learner_player", 0, 1), ("policy_seed", 0, 2**63-1)):
            c._integer(row.get(name), low=low, high=high)
        _opponent_token(self.opponent_identity, row["learner_player"])
        require(Path(self.folder).is_absolute() and Path(self.plan_path).is_absolute(), "Use absolute capture and plan paths")
        self.recheck()

    def recheck(self):
        require(digest(self.plan_path) == self.plan_sha256, "Approved training plan changed")
        plan = native.strict_json(Path(self.plan_path).read_bytes())
        require(type(self.schedule_index) is int and 0 <= self.schedule_index < len(plan["schedule"])
                and canonical(plan["schedule"][self.schedule_index]) == self.row_json, "Training schedule slot changed")
        checkpoint = json.loads(self.checkpoint_file_json)
        require(type(checkpoint) is dict and set(checkpoint) == {"path", "sha256", "bytes"}
                and Path(checkpoint["path"]).is_absolute()
                and canonical({"path": checkpoint["path"], **native.file_identity(Path(checkpoint["path"]))}) == self.checkpoint_file_json,
                "Frozen behavior checkpoint file changed")
        require(type(json.loads(self.source_json)) is dict and type(json.loads(self.native_identity_json)) is dict,
                "Source and native identities must be explicit objects")
        # The runner's complete implementation inventory includes this new gate.
        # It is checked here as well as by the supervising worker.
        root = Path(__file__).resolve().parents[2]
        implementation = plan.get("implementation")
        require(type(implementation) is dict and bool(implementation), "Plan must bind its complete implementation")
        for name, sha in implementation.items():
            require(digest(native.inside(root, name)) == sha, "Prepared implementation changed: " + name)
        require(canonical(plan.get("source")) == self.source_json, "Binding differs from planned learner source")
        require(canonical(plan.get("native")) == self.native_identity_json,
                "Binding differs from planned native runtime identity")
        require(canonical(plan.get("checkpoint")) == self.checkpoint_file_json,
                "Training must start from the planned checkpoint file")
        cfg = plan["configuration"]
        if self.max_choices == 5120:
            require(plan["schema"] == "heroes-builtin-training-v1-plan"
                    and cfg.get("reserved_dialog_choices") == 160
                    and cfg.get("allow_verified_training_prefix") is False,
                    "Expanded choices require an explicit complete continuation batch")
        require(type(cfg.get("max_days")) is int and cfg["max_days"] == self.max_days
                and type(cfg.get("ordinary_choice_limit")) is int and cfg["ordinary_choice_limit"] == self.nominal_choices
                and type(cfg.get("max_total_choices")) is int and cfg["max_total_choices"] == self.max_choices,
                "Binding differs from the planned collection limits")

    def to_dict(self):
        return {"schema": BINDING_SCHEMA, **{name: getattr(self, name) for name in self.__dataclass_fields__}}


@dataclass(frozen=True)
class TrainingCapture:
    """Fresh collection intent around the unchanged, ineligible own record."""
    experience: BuiltinCollectedEpisode
    binding: TrainingBinding

    @property
    def history(self): return self.experience.history
    @property
    def decisions(self): return self.experience.decisions
    @property
    def estimates(self): return self.experience.estimates
    @property
    def manifest(self): return self.experience.manifest
    @property
    def rollout(self): return self.experience

    def to_dict(self):
        return {"schema": CAPTURE_SCHEMA, "purpose": "training", "optimization_eligible": False,
                "binding": self.binding.to_dict(), "experience": self.experience.to_dict()}


class _ParallelCaptureLease:
    """One live child may transfer one freshly planned capture, never a receipt.

    The coordinator creates this capability before the episode directory exists.
    Completion is read directly from the exclusively attached child's stdout;
    an artifact on disk, a deserialized lease, or a caller-supplied completion
    cannot recreate this capability. Ordinary independent verification still
    runs in the coordinator before it can mint an update receipt.
    """
    def __init__(self, binding):
        self.binding, self.token, self.process = binding, secrets.token_hex(32), None
        self.consumed = False


def _begin_parallel_capture(binding):
    require(type(binding) is TrainingBinding and _BINDINGS.get(id(binding)) is binding,
            "Parallel collection needs this coordinator's fresh prelaunch binding")
    binding.validate()
    require(not Path(binding.folder).exists(), "Parallel capture directory already exists")
    lease = _ParallelCaptureLease(binding)
    _PARALLEL_CAPTURES[id(lease)] = lease
    return lease


def _attach_parallel_process(lease, process):
    require(_PARALLEL_CAPTURES.get(id(lease)) is lease and not lease.consumed
            and lease.process is None, "Parallel capture lease is not fresh")
    require(getattr(process, '_deadline_guarded', False) is True
            and getattr(process, '_deadline_owner_fd', None) is not None
            and process.stdout is not None and process.poll() is None,
            "Parallel capture must own a live guarded child and its private result pipe")
    lease.process = process


def _cancel_parallel_capture(lease):
    require(_PARALLEL_CAPTURES.get(id(lease)) is lease, "Unknown parallel capture lease")
    lease.consumed = True
    _PARALLEL_CAPTURES.pop(id(lease), None)


def _complete_parallel_capture(lease):
    """Consume live launch provenance, then reconstruct the bound fresh capture."""
    require(_PARALLEL_CAPTURES.get(id(lease)) is lease and not lease.consumed
            and lease.process is not None, "No fresh live parallel capture lease")
    lease.consumed = True
    _PARALLEL_CAPTURES.pop(id(lease), None)
    process, binding = lease.process, lease.binding
    require(process.poll() == 0, "Parallel collector did not finish successfully")
    # This is the actual private pipe from the process attached before launch
    # completion; no completion/result-file argument is accepted by this gate.
    message = native.strict_json(process.stdout.read(65537))
    require(type(message) is dict and message.get('token') == lease.token
            and message.get('schedule_index') == binding.schedule_index
            and message.get('plan_sha256') == binding.plan_sha256
            and message.get('status') == 'passed', "Parallel collector completion differs from its live lease")
    binding.recheck()
    folder = Path(binding.folder)
    path = folder / 'builtin-training-rollout.json'
    require(message.get('capture') == native.file_identity(path),
            "Parallel capture bytes changed after child completion")
    raw = native.strict_json(path.read_bytes())
    require(raw.get('schema') == CAPTURE_SCHEMA and raw.get('purpose') == 'training'
            and raw.get('optimization_eligible') is False
            and raw.get('binding') == binding.to_dict(), "Parallel capture has another binding")
    capture = TrainingCapture(load_builtin_episode(raw['experience']), binding)
    _check_capture(capture, binding)
    require(canonical(raw) == canonical(capture.to_dict()), "Parallel capture contains extra or changed fields")
    _CAPTURES[id(capture)] = capture
    return capture, message


class TrainingCollector(BuiltinCollector):
    """One fresh scheduled capture; still no eligibility before independent proof."""
    def __init__(self, model, binding, *, interface=c.DIALOGS_ENVIRONMENT):
        require(type(binding) is TrainingBinding and _BINDINGS.get(id(binding)) is binding,
                "Training requires this worker's prelaunch binding")
        binding.validate()
        require(weights_sha256(model) == binding.weights_sha256, "Training collector has different behavior weights")
        require(json.loads(binding.source_json).get("interface", c.DIALOGS_ENVIRONMENT) == interface,
                "Collector interface differs from planned source")
        self.binding = binding
        super().__init__(model, binding.checkpoint, learner_player=binding.row["learner_player"],
                         opponent_identity=binding.opponent_identity, policy_seed=binding.row["policy_seed"], interface=interface)

    def act(self, request):
        used = len(self.session._choices) if self.session is not None else 0
        require(used < self.binding.max_choices, "Hard total training choice cap reached")
        require(c.decode_frame(request.record).day <= self.binding.max_days,
                "Training policy cannot sample beyond the day limit")
        return super().act(request)

    def run_episode(self, env, *, episode, seed, max_requests=None):
        binding = self.binding
        binding.recheck()
        row = binding.row
        require(type(episode) is int and type(seed) is int and (episode, seed) == (row["episode"], row["engine_seed"])
                and (max_requests is None or max_requests == binding.nominal_choices), "Training collection differs from its slot")
        folder = Path(env.output).resolve() / f"episode-{episode:06d}"
        require(str(folder) == binding.folder and not folder.exists(), "Training capture directory is not fresh")
        require(canonical(env.launcher.identity()) == binding.native_identity_json, "Prelaunch native identity differs")
        return super().run_episode(env, episode=episode, seed=seed, max_requests=binding.nominal_choices)

    def finish(self, end, *, manifest_path):
        experience = super().finish(end, manifest_path=manifest_path)
        result = TrainingCapture(experience, self.binding)
        _check_capture(result, self.binding)
        folder = Path(manifest_path).resolve().parent
        require(str(folder) == self.binding.folder, "Training finalization is outside its fresh directory")
        with (folder / "builtin-training-rollout.json").open("x") as stream:
            # json.dumps writes the same text as json.dump with the C encoder.
            stream.write(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False))
            stream.write("\n")
        _CAPTURES[id(result)] = result
        return result


def _check_capture(capture, binding):
    require(type(capture) is TrainingCapture and capture.binding is binding
            and type(capture.experience) is BuiltinCollectedEpisode, "Expected this fresh training capture")
    item, manifest, row = capture.experience, capture.manifest, binding.row
    require(item.player == row["learner_player"] and item.policy_seed == row["policy_seed"]
            and item.checkpoint == binding.checkpoint and item.weights_sha256 == binding.weights_sha256
            and canonical(item.opponent_identity) == binding.opponent_json, "Training policy/seat/seed/opponent binding differs")
    expected_native = json.loads(binding.native_identity_json)
    plan = native.strict_json(Path(binding.plan_path).read_bytes())
    _check_policy_support(item, plan)
    require(manifest["episode"] == str(row["episode"]) and type(manifest["seed"]) is int
            and manifest["seed"] == row["engine_seed"]
            and canonical({key: manifest["identity"].get(key) for key in expected_native}) == binding.native_identity_json
            and canonical(manifest["identity"].get("native_runtime_sha256")) == canonical(plan["runtime"]),
            "Training capture source/native identity or schedule differs")
    require(len(item.decisions) <= binding.max_choices and all(item.history[d.decision_seq].day <= binding.max_days
            for d in item.decisions), "Training capture exceeds sampled-choice/day budgets")
    if len(item.decisions) > binding.nominal_choices:
        drain = manifest.get("dialog_cutoff_drain")
        require(type(drain) is dict and drain["trigger"]["actions_sent"] == binding.nominal_choices,
                "Additional training choices require the declared real dialog drain")


def _check_policy_support(episode, plan):
    """Recorded behavior must use the support intervention declared by its plan."""
    from .contextual_rollout import behavior_guard_version
    from .interaction_guard import VERSION
    expected = plan.get('policy_support')
    require(expected is None or expected == VERSION, 'Unknown planned policy support')
    if episode.decisions:
        require(behavior_guard_version(episode.decisions) == expected,
                'Captured policy support differs from the immutable plan')


@dataclass(frozen=True)
class BuiltinTerminalClosure:
    consumed_seq: int
    superseded_seq: int
    terminal_seq: int
    evidence_ref: str


@dataclass(frozen=True)
class BuiltinSample:
    decision: object
    result_seq: int | None
    successor_seq: int | None
    reward: int
    bootstrap_value: float | None
    eligible: bool
    terminal_closure: BuiltinTerminalClosure | None = None


def _samples(experience, evidence_ref):
    """Derive targets from unchanged own frames and separately verified closure."""
    frames, manifest = experience.history, experience.manifest
    boundary = {"completed": "natural", "truncated": "cutoff"}.get(manifest["status"])
    require(boundary is not None, "Technical failures have no training targets")
    history = c.PlayerHistory(frames[0].episode, experience.player, schema=evidence.manifest_profile(manifest).history_schema)
    for frame in frames: history.append(frame)
    evidence.validate_manifest_history(history, manifest)
    values = {item.decision_seq: item.value for item in experience.estimates}
    records = {item.decision_seq: item for item in experience.decisions}
    consumed = {f.decision_seq for f in frames if f.kind == "consumed"}
    withdrawn = {f.decision_seq for f in frames if f.kind == "superseded" and f.decision_seq not in consumed}
    decisions = [f.seq for f in frames if f.kind == "decision" and f.seq not in withdrawn]
    completed = [(f.decision_seq, f.seq, None) for f in frames if f.kind == "result"]
    if boundary == "natural" and consumed:
        last = max(consumed)
        if last == max(f.seq for f in frames if f.kind == "decision") and not any(seq == last for seq, _, _ in completed):
            related = [f for f in frames if f.decision_seq == last]
            submissions = [f for f in related if f.kind == "submitted"]
            superseded = [f for f in related if f.kind == "superseded"]
            consumption = [f for f in related if f.kind == "consumed"]
            terminals = [f for f in frames if f.kind == "terminal"]
            require(len(submissions) == len(superseded) == len(consumption) == len(terminals) == 1
                    and superseded[0].reason == "terminal"
                    and last < consumption[0].seq < superseded[0].seq < terminals[0].seq,
                    "Final consumed action lacks an exact natural terminal closure")
            completed.append((last, None, BuiltinTerminalClosure(consumption[0].seq,
                superseded[0].seq, terminals[0].seq, evidence_ref)))
    require(completed and consumed == {seq for seq, _, _ in completed},
            "Training cannot omit consumed actions or bridge an incomplete trajectory")
    require([seq for seq, _, _ in completed] == sorted(set(seq for seq, _, _ in completed)),
            "Training results must follow chronological own decisions")
    samples = []
    for index, (seq, result, closure) in enumerate(completed):
        terminal = boundary == "natural" and index == len(completed) - 1
        successor = None if terminal else next((value for value in decisions if result is not None and value > result), None)
        require(seq in records and (terminal or successor in values), "Training action lacks its actual same-player successor")
        reward = (1 if history.terminal == "win" else -1) if terminal else 0
        samples.append(BuiltinSample(records[seq], result, successor, reward,
                                     None if terminal else values[successor], True, closure))
    for first, second in zip(samples, samples[1:]):
        require(first.successor_seq == second.decision.decision_seq and first.reward == 0
                and first.bootstrap_value == second.decision.value, "Training successor is not the next own sample")
    if boundary == "cutoff":
        require(frames[-1].kind == "cutoff" and samples[-1].successor_seq == frames[-1].decision_seq,
                "Cutoff bootstrap must be the actual held own decision")
    return tuple(samples)


@dataclass(frozen=True)
class BuiltinTrainingRollout:
    experience: BuiltinCollectedEpisode
    binding: TrainingBinding
    evidence_ref: str
    samples: tuple[BuiltinSample, ...]

    def __post_init__(self):
        require(type(self.experience) is BuiltinCollectedEpisode and type(self.binding) is TrainingBinding
                and type(self.samples) is tuple and all(type(s) is BuiltinSample for s in self.samples),
                "Expected strict single-learner training records")
        require(self.samples == _samples(self.experience, self.evidence_ref), "Training targets differ from verified own history")

    @property
    def history(self): return self.experience.history
    @property
    def decisions(self): return self.experience.decisions
    @property
    def boundary(self): return "natural" if self.experience.manifest["status"] == "completed" else "cutoff"
    @property
    def player(self): return self.experience.player
    @property
    def weights_sha256(self): return self.experience.weights_sha256
    @property
    def behavior_checkpoint(self): return self.experience.checkpoint
    @property
    def opponent_checkpoint(self): return _opponent_token(self.experience.opponent_identity, self.player)
    opponent_weights_sha256 = None
    purpose, origin, role = "training", "native", "learner"
    optimization_eligible = True


@dataclass(frozen=True)
class VerifiedTrainingReceipt:
    binding: TrainingBinding
    rollout: BuiltinTrainingRollout
    files_json: str
    replay_binary: str
    policy_reconstruction_json: str
    history_reconstruction_json: str

    def to_dict(self):
        return {"schema": RECEIPT_SCHEMA, "binding": self.binding.to_dict(), "files": json.loads(self.files_json),
                "replay_binary": self.replay_binary, "policy_reconstruction": json.loads(self.policy_reconstruction_json),
                "history_reconstruction": json.loads(self.history_reconstruction_json),
                "rollout_schema": ROLLOUT_SCHEMA, "samples": len(self.rollout.samples),
                "opponent_history": None, "opponent_policy": None, "admission_from_file": False}


def _write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(canonical(value) + "\n")


def _capture_file_bindings(folder, manifest):
    declaration = manifest["players"][evidence.COLORS[manifest["learner_player"]]]
    names = {"manifest.json", "builtin-finalization.json", "builtin-training-rollout.json", "logs/process.log",
        "engine/experiment-metadata.json", f"agents/{evidence.COLORS[manifest['learner_player']]}/bridge.jsonl",
        declaration[evidence.manifest_profile(manifest).history_key]["path"], declaration["live_evidence"]["path"], *manifest["replay"]["files"]}
    return {name: native.file_identity(native.inside(folder, name)) for name in sorted(names)}


def _check_persisted(folder, collected, model, manifest):
    """The persisted rollout and manifest checks, in their original order; returns the rebuilt episode."""
    raw = native.strict_json((folder / "builtin-training-rollout.json").read_bytes())
    require(canonical(raw) == canonical(collected.to_dict()), "Fresh training artifact changed after capture")
    rebuilt = load_builtin_episode(raw["experience"])
    require(rebuilt == collected.experience and rebuilt.weights_sha256 == weights_sha256(model)
            and json.loads(rebuilt.model_config_json) == model.config.to_dict(), "Behavior differs from current frozen model")
    require(canonical(manifest) == canonical(rebuilt.manifest) and manifest["status"] in ("completed", "truncated"),
            "Only an intact naturally ended or verified cutoff capture can be admitted")
    return rebuilt


def _check_rollout(folder, collected, binding, model, manifest, verify_history, reconstruct_behavior):
    rebuilt = _check_persisted(folder, collected, model, manifest)
    final = evidence.read_finalization(folder, manifest)
    history_report = verify_history(folder, manifest, final)
    require(history_report.get("verified") is True and history_report.get("opponent_history") is None,
            "Independent single-player history reconstruction failed")
    policy_report = reconstruct_behavior(rebuilt.history, rebuilt.decisions, rebuilt.estimates, model,
                                         binding.row["policy_seed"], schema=evidence.manifest_profile(manifest).history_schema)
    require(policy_report.get("maximum_absolute_difference") == 0 and policy_report.get("weights_unchanged") is True
            and policy_report.get("private_memory_reset") is True
            and policy_report.get("actions_resampled") == len(rebuilt.decisions)
            and policy_report.get("values_recomputed") == len(rebuilt.estimates), "Exact independent behavior reconstruction failed")
    _write_new(folder / "training-history-reconstruction.json", history_report)
    _write_new(folder / "training-policy-reconstruction.json", policy_report)
    return history_report, policy_report, rebuilt, final


def verify_training_capture(folder, collected, binding, model, *, replay_binary, data, remaining):
    """Mint an in-memory receipt only after independent reconstruction and replay."""
    from experiments.headless.background_check import BackgroundCheck
    from experiments.headless.builtin_replay_history import verify_history
    from experiments.headless import builtin_replay
    from validation.probes.validate_hero_policy import reconstruct_behavior
    _time_left(remaining)
    require(_BINDINGS.get(id(binding)) is binding and _CAPTURES.get(id(collected)) is collected,
            "Only this worker's fresh training collector can request admission")
    folder = Path(folder).resolve()
    require(str(folder) == binding.folder, "Admission folder differs from fresh collection")
    binding.recheck()
    _check_capture(collected, binding)
    manifest = native.strict_json((folder / "manifest.json").read_bytes())
    # Bind the capture first: the independent native replay reads it in its own interpreter
    # while this process checks the rollout, history and policy; every check below still runs.
    # The capture is bound from a manifest equal to the fresh capture's own, which the persisted
    # checks require anyway. Otherwise, or if binding fails, it is bound in the serial order,
    # after those checks, so any error raised is the one the serial order raised.
    try:
        require(canonical(manifest) == canonical(collected.experience.manifest)
                and manifest["status"] in ("completed", "truncated"), "Manifest differs from the fresh capture")
        original_files = _capture_file_bindings(folder, manifest)
    except Exception:
        _check_persisted(folder, collected, model, manifest)
        original_files = _capture_file_bindings(folder, manifest)
    replaying = BackgroundCheck(builtin_replay.verify_episode, folder, binary=replay_binary, data=data,
                                output=folder / "training-verification", timeout=_time_left(remaining))
    try:
        replaying.start()
        history_report, policy_report, rebuilt, final = _check_rollout(folder, collected, binding, model, manifest,
                                                                       verify_history, reconstruct_behavior)
        replay = replaying.result(timeout=_time_left(remaining))
    finally:
        replaying.stop()
    require(replay.get("schema") == builtin_replay.REPORT_SCHEMA
            and replay.get("identity_check", {}).get("mode") == "exact_capture"
            and "diagnostic_verified" not in replay,
            "Training requires exact-runtime native replay, never supplemental diagnostic compatibility")
    require(replay.get("verified") is True,
            "Exact-runtime native replay failed: " + str(replay.get("error") or "verification did not pass"))
    require(replay.get("history") == history_report, "Native replay history differs from independent reconstruction")
    require(canonical(native.strict_json((folder / "training-verification/builtin-replay-verification.json").read_bytes()))
            == canonical(replay), "Native verification report differs from its persisted evidence")
    native_document = native.strict_json((folder / "training-verification/native-report.json").read_bytes())
    require(canonical(native_document) == canonical({key: value for key, value in replay["native"].items()
            if key not in ("returncode", "seconds")}), "Native report changed after verifier execution")
    builtin_replay.validate_native_report(replay["native"], replay["wire"], final, cutoff=manifest["status"] == "truncated")
    binding.recheck()
    require(weights_sha256(model) == binding.weights_sha256, "Frozen learner changed during verification")
    _time_left(remaining)
    require(_capture_file_bindings(folder, manifest) == original_files, "Capture evidence changed during verification")
    names = {"training-history-reconstruction.json", "training-policy-reconstruction.json",
        "training-verification/builtin-replay-verification.json", "training-verification/native-report.json"}
    files = {**original_files, **{name: native.file_identity(native.inside(folder, name)) for name in sorted(names)}}
    reference = "training-verification/builtin-replay-verification.json#sha256=" + files["training-verification/builtin-replay-verification.json"]["sha256"]
    rollout = BuiltinTrainingRollout(rebuilt, binding, reference, _samples(rebuilt, reference))
    receipt = VerifiedTrainingReceipt(binding, rollout, canonical(files), str(Path(replay_binary).resolve()),
        canonical(policy_report), canonical(history_report))
    _write_new(folder / "training-receipt.json", receipt.to_dict())
    _RECEIPTS[id(receipt)] = receipt
    return receipt


def _admit_builtin(receipts, model, *, plan_sha256):
    from experiments.headless.builtin_replay import verify_identity
    require(type(receipts) is tuple and 1 <= len(receipts) <= 8, "One batch needs its complete planned fresh captures")
    require(all(type(receipt) is VerifiedTrainingReceipt and _RECEIPTS.get(id(receipt)) is receipt
                and id(receipt) not in _USED_RECEIPTS for receipt in receipts), "Receipt is not fresh or was already used")
    require(len({id(receipt) for receipt in receipts}) == len(receipts), "Duplicate training receipt")
    plan = native.strict_json(Path(receipts[0].binding.plan_path).read_bytes())
    collection = _parallel_collection_contract(plan)
    if collection is not None:
        require(plan.get('schema') == 'heroes-builtin-training-v1-plan'
                and len(receipts) == collection['batch_size'],
                "Parallel collection requires its complete explicit batch")
    else:
        require(len(receipts) <= 4, "Eight captures require an explicit complete parallel collection plan")
    expected = [index for index, row in enumerate(plan["schedule"]) if row.get("phase") == "training"]
    require([receipt.binding.schedule_index for receipt in receipts] == expected[:len(receipts)]
            and (len(receipts) == len(expected)
                 or plan["configuration"].get("allow_verified_training_prefix") is True),
            "Update requires its explicitly allowed contiguous verified training prefix")
    current = weights_sha256(model)
    identities = set()
    for receipt in receipts:
        binding, rollout = receipt.binding, receipt.rollout
        require(binding.plan_sha256 == plan_sha256 and type(rollout) is BuiltinTrainingRollout
                and rollout.binding is binding and rollout.weights_sha256 == current,
                "Training receipt is off-policy or belongs to another plan")
        binding.recheck()
        folder = Path(binding.folder)
        for name, identity in json.loads(receipt.files_json).items():
            require(native.file_identity(native.inside(folder, name)) == identity, "Admitted episode evidence changed: " + name)
        require(native.strict_json((folder / "training-receipt.json").read_bytes()) == receipt.to_dict(), "Persisted receipt changed")
        verify_identity(rollout.experience.manifest, Path(receipt.replay_binary))
        require(rollout.samples == _samples(rollout.experience, rollout.evidence_ref), "Training targets changed after verification")
        identities.add((binding.checkpoint, binding.weights_sha256, binding.opponent_identity["binary_sha256"],
                        binding.source_json, binding.native_identity_json))
    require(len(identities) == 1, "A batched update cannot mix frozen policy, source or native opponent identities")
    return tuple(receipt.rollout for receipt in receipts)


def _parallel_collection_contract(plan):
    """Preserve four-game PPO batches; retain the explicit eight-game diagnostic."""
    cfg = plan.get('configuration', {})
    selected = plan.get('execution', {}).get('collection_workers', 1)
    require(type(selected) is int and selected in (1, 4), 'Unknown collection worker count')
    if 'parallel_collection' in cfg:
        value = cfg['parallel_collection']
        require(selected == 1, 'Conflicting collection declarations')
        require(type(value) is dict and set(value) == {'batch_size', 'workers'}
                and type(value['batch_size']) is int and value['batch_size'] in (4, 8)
                and type(value['workers']) is int and value['workers'] in (1, value['batch_size']),
                'Declare a complete four- or eight-game collection batch')
    elif 'collection_workers' in plan.get('execution', {}):
        value = {'batch_size': 4, 'workers': selected}
    else:
        return None
    rows = plan.get('schedule')
    require(type(rows) is list and len(rows) == value['batch_size']
            and cfg.get('allow_verified_training_prefix') is False,
            'Parallel collection cannot admit a partial planned batch')
    phase = 'training' if plan.get('schema') == 'heroes-builtin-training-v1-plan' else 'evaluation'
    require(plan.get('schema') in ('heroes-builtin-training-v1-plan',
                                   'heroes-builtin-checkpoint-evaluation-v1-plan')
            and all(type(row) is dict and row.get('phase') == phase for row in rows),
            'Parallel collection needs one declared phase')
    return dict(value)


def _parallel_eight(plan):
    value = _parallel_collection_contract(plan)
    return (value is not None and value['batch_size'] == 8
            and plan.get('schema') == 'heroes-builtin-training-v1-plan')


def native_update(model, optimizer, receipts, *, plan_sha256, config, generator, remaining, max_optimizer_steps=32):
    """One complete on-policy batch; only this worker's verified receipts admit."""
    _time_left(remaining)
    require(type(config) is PPOConfig and config.gamma == config.gae_lambda == 1
            and config.epochs == 1 and config.decisions_per_window == 16 and config.windows_per_minibatch == 2,
            "Pilot PPO must retain the predeclared undiscounted one-epoch windows")
    require(type(max_optimizer_steps) is int and 1 <= max_optimizer_steps <= 1280
            and type(optimizer) is torch.optim.Adam
            and all(group["lr"] == 0.0003 for group in optimizer.param_groups), "Pilot requires bounded Adam at 3e-4")
    rollouts = _admit_builtin(receipts, model, plan_sha256=plan_sha256)
    plan = native.strict_json(Path(receipts[0].binding.plan_path).read_bytes())
    parallel = _parallel_eight(plan)
    require(max_optimizer_steps <= (1280 if parallel else 640),
            "Expanded optimizer cap requires an explicit eight-game collection plan")
    require(canonical(config.to_dict()) == canonical(plan["configuration"]["ppo"]),
            "PPO configuration differs from the approved plan")
    require(max_optimizer_steps == plan["configuration"]["max_optimizer_steps"],
            "Optimizer cap differs from the immutable plan")
    if max_optimizer_steps > 32:
        require(plan["schema"] == "heroes-builtin-training-v1-plan"
                and plan["configuration"]["allow_verified_training_prefix"] is False
                and len(rollouts) == len(plan["schedule"]) == (8 if parallel else 4)
                and plan["configuration"]["ordinary_choice_limit"] == 4960
                and plan["configuration"]["reserved_dialog_choices"] == 160
                and plan["configuration"]["max_total_choices"] == 5120,
                "Expanded cap requires the complete declared continuation batch")
    windows = sum((len(rollout.samples) + 15) // 16 for rollout in rollouts)
    require((windows + 1) // 2 <= max_optimizer_steps, "Complete update exceeds its optimizer-step cap")
    _USED_RECEIPTS.update(id(receipt) for receipt in receipts)
    result = transactional_update(model, optimizer, rollouts, config=config, generator=generator,
        remaining=lambda: _time_left(remaining), max_optimizer_steps=max_optimizer_steps, enable_training=True)
    return replace(result, purpose=PROVENANCE), rollouts


class UpdateRolledBack(RuntimeError):
    def __init__(self, error, completed_steps):
        super().__init__(f"{type(error).__name__}: {error}; model, Adam and all RNG state restored")
        self.completed_steps = completed_steps


def transactional_update(model, optimizer, rollouts, *, config, generator, remaining, max_optimizer_steps,
                         enable_training=False):
    """Restore the entire attempted update on failure, including interrupted steps."""
    parameters = tuple(model.parameters())
    state, optimizer_state = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    gradients = tuple(None if p.grad is None else p.grad.detach().clone() for p in parameters)
    requires_grad = tuple(p.requires_grad for p in parameters)
    modes = tuple((module, module.training) for module in model.modules())
    torch_rng, python_rng, minibatch_rng = torch.get_rng_state().clone(), random.getstate(), generator.get_state().clone()
    completed_steps = 0

    def completed():
        nonlocal completed_steps
        completed_steps += 1

    try:
        remaining()
        if enable_training:
            model.requires_grad_(True)
        result = _update(model, optimizer, rollouts, config=config, generator=generator,
            before_step=remaining, after_step=completed, max_optimizer_steps=max_optimizer_steps)
        remaining()
        return result
    except BaseException as error:
        model.load_state_dict(state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        for parameter, gradient, flag in zip(parameters, gradients, requires_grad):
            parameter.requires_grad_(flag)
            parameter.grad = gradient
        for module, mode in modes:
            module.training = mode
        torch.set_rng_state(torch_rng)
        random.setstate(python_rng)
        generator.set_state(minibatch_rng)
        raise UpdateRolledBack(error, completed_steps) from error
    finally:
        if enable_training:
            for parameter, flag in zip(parameters, requires_grad):
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
