"""Controller contracts use disposable files and mocks, never native gameplay."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import run_selfplay_training as controller


def configuration():
    return dict(interface=controller.INTERFACE, adam={'lr': 0.0003}, ppo=dict(controller.PPO),
                max_days=365, ordinary_choice_limit=4960, reserved_dialog_choices=160,
                max_total_choices=5120, max_optimizer_steps=640, attempt_wall_seconds=600,
                allow_verified_training_prefix=False, training_games=4, cleanup_reserve_seconds=30,
                phase_deadlines_seconds={'training': 2400, 'update': 10740, 'evaluation': 3540})


class SelfPlayControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='heroes-selfplay-controller-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def artifact(self, name, value):
        path = self.root / name
        controller.write(path, value, new=True)
        return controller.binding(path)

    def test_explicit_subcommand_required_without_launch(self):
        with patch.object(controller, 'supervise') as launch:
            with self.assertRaises(SystemExit) as error:
                controller.main([])
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_100_physical_games_are_25_balanced_batches_with_independent_policy_seeds(self):
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            reservation = controller.reserve_schedule(self.root / 'training', 'training', 100)
        rows = [row for offset in range(0, 100, 4) for row in controller.schedule(reservation, offset, 4)]
        self.assertEqual(len(rows), 100)
        self.assertEqual([sum(row['learner_player'] == player for row in rows) for player in (0, 1)], [50, 50])
        self.assertEqual(len({row['engine_seed'] for row in rows}), 100)
        self.assertEqual(len({seed for row in rows for seed in row['policy_seeds']}), 200)
        for offset in range(0, 100, 4):
            self.assertEqual([row['episode'] for row in rows[offset:offset + 4]], [1, 2, 3, 4])
            self.assertEqual([row['learner_player'] for row in rows[offset:offset + 4]], [0, 1, 0, 1])

    def test_unused_reserved_seeds_remain_consumed_and_diagnostics_are_disjoint(self):
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            old = controller.reserve_schedule(self.root / 'never_started', 'training', 100)
            new = controller.reserve_schedule(self.root / 'fresh', 'training', 100)
            diagnostic = controller.reserve_schedule(self.root / 'diagnostic', 'diagnostic', 8)
            self.assertEqual(new['offset'], 100)
            values = [controller.schedule(item, 0, item['count']) for item in (old, new, diagnostic)]
            seeds = [{row['engine_seed'] for row in rows} for rows in values]
            self.assertFalse(seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2])
            before = (controller.BASE / 'seed-reservations.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'already reserved'):
                controller.reserve_schedule(self.root / 'fresh', 'training', 100)
            self.assertEqual((controller.BASE / 'seed-reservations.json').read_bytes(), before)

    def test_continuation_reserves_92_fresh_seeds_after_all_100_consumed(self):
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            original = controller.reserve_schedule(self.root / 'original', 'training', 100)
            before = (controller.BASE / 'seed-reservations.json').read_bytes()
            preview = controller.seed_preview('training', 92)
            self.assertEqual((controller.BASE / 'seed-reservations.json').read_bytes(), before)
            self.assertFalse(preview['reserved'])
            self.assertEqual(preview['consumed_before'], 100)
            resumed = controller.reserve_schedule(self.root / 'resumed', 'training', 92)
            self.assertEqual(resumed['count'], 92)
            self.assertEqual(resumed['engine_base'], original['engine_base'] + 100)
            self.assertEqual(resumed['policy_base'], original['policy_base'] + 200)
            self.assertEqual(len(controller.schedule(resumed, 0, 92)), 92)

    def test_continuation_reservation_never_recreates_a_missing_verified_ledger(self):
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            controller.reserve_schedule(self.root / 'original', 'training', 100)
            path = controller.BASE / 'seed-reservations.json'
            expected = controller.binding(path)
            before = path.read_bytes(); path.unlink()
            with self.assertRaises(FileNotFoundError):
                controller.reserve_schedule(self.root / 'resumed', 'training', 92, expected_ledger=expected)
            self.assertFalse(path.exists())
            path.write_bytes(before)
            changed = controller.read(path); changed['reservations'] = []
            controller.write(path, changed)
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                controller.reserve_schedule(self.root / 'resumed', 'training', 92, expected_ledger=expected)
            self.assertEqual(controller.read(path), changed)

    def lineage_fixture(self):
        original = (self.root / 'original').resolve(); original.mkdir()
        source = dict(checkout={'commit': controller.CONTINUATION_BASE}, source={}, implementation={})
        parent = dict(policy=self.artifact('parent.pt', {'checkpoint': 1728}),
                      report=self.artifact('parent-report.json', {'cumulative_progress': {'updates': 1728, 'rollouts': 6912}}))
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            reservation = controller.reserve_schedule(original, 'training', 100)
        launch = dict(output=str(original), campaign_id=original.name, source=source, mode='training',
                      original_parent=parent, reservation=reservation, configuration=configuration(),
                      protected={}, native_manifest={}, runtime={}, native={})
        launch_binding = self.artifact('original/launch.json', launch)
        initial = dict(progress=dict(updates=1728, rollouts=6912, decisions=978803),
                       optimizer_steps=32814, sampler={'optimizer_steps': 32814},
                       checkpoint='initial1728', weights_sha256='initial-weights')
        patcher = patch.object(controller, 'initial_checkpoint_state', return_value=initial)
        patcher.start(); self.addCleanup(patcher.stop)
        previous_policy, previous_checkpoint, previous_weights = parent['policy'], initial['checkpoint'], initial['weights_sha256']
        batches = []
        for index in range(2):
            plan = dict(launch=launch_binding, configuration=configuration(), source={}, implementation={},
                learner_checkpoint=previous_checkpoint, learner_weights_sha256=previous_weights,
                opponent_weights_sha256=previous_weights, opponent_checkpoint=parent['policy'],
                checkpoint_source={'opponent_checkpoint': previous_policy},
                opponent_checkpoint_source={'opponent_checkpoint': previous_policy})
            plan_binding = self.artifact(f'original/batch-{index:03d}/plan.json', plan)
            pair = dict(progress=dict(updates=1729 + index, rollouts=6916 + index * 4, decisions=978813 + index * 10),
                policy=self.artifact(f'original/batch-{index:03d}/policy.pt', {'policy': index}),
                resume=self.artifact(f'original/batch-{index:03d}/resume.pt', {'adam': index}),
                exact_model_adam_rng_progress_pool=True, checkpoint=f'selfplay{1729 + index}', weights_sha256=f'weights{index}',
                sampler_state=dict(next_batch_index=index + 1, consumed_schedule_offset=(index + 1) * 4,
                    optimizer_steps=32815 + index, batch_optimizer_steps=1, original_parent_sampler=initial['sampler']),
                metadata=dict(purpose='training', initial_checkpoint=parent['policy'], campaign_id=original.name,
                    plan_sha256=plan_binding['sha256'], opponent_checkpoint=parent['policy']))
            attempts = [dict(status='passed', replay_verified=True, both_policies_reconstructed=True) for _ in range(4)]
            batches.append(dict(index=index, status='checkpoint_verified', plan=plan_binding,
                                attempts=attempts, after_checkpoint=pair, ppo={'samples': 10, 'optimizer_steps': 1}))
            previous_policy, previous_checkpoint, previous_weights = pair['policy'], pair['checkpoint'], pair['weights_sha256']
        batches.append(dict(index=2, status='updating', attempts=deepcopy(batches[0]['attempts'])))
        pointer = dict(schema='heroes-selfplay-latest-v1', purpose='training', campaign_id=original.name,
            batch=1, admitted_games=8, checkpoint=pair, plan=plan_binding, cumulative_progress=pair['progress'])
        self.artifact('original/latest-checkpoint.json', pointer)
        self.artifact('original/report.json', dict(source=source, mode='training', status='running', admitted_games=8, batches=batches))
        self.artifact('original/shutdown.json', dict(owned_cleanup_verified=True, protected_inputs_unchanged=True, returncode=-9))
        controller.finalize_stopped_run(original, launch)
        path = self.artifact('lineage.json', dict(schema='heroes-selfplay-lineage-v1', target_games=100, runs=[launch_binding]))['path']
        return Path(path), original, pointer

    def test_lineage_uses_published_1730_and_ignores_interrupted_third_update(self):
        path, original, pointer = self.lineage_fixture()
        self.artifact('original/batch-002/unpublished-policy.pt', {'must_not_load': 1731})
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            result = controller.inspect_lineage(path)
        self.assertEqual(result['admitted_games'], 8)
        self.assertEqual(result['remaining_games'], 92)
        self.assertEqual(result['selected']['checkpoint'], pointer['checkpoint'])
        self.assertEqual(controller.read(original / 'stop-finalization.json')['ignored_unpublished_batches'], [2])

    def test_lineage_rejects_a_candidate_promoted_outside_the_published_chain(self):
        path, original, pointer = self.lineage_fixture()
        pointer.update(batch=2, admitted_games=12)
        controller.write(original / 'latest-checkpoint.json', pointer)
        with patch.object(controller, 'BASE', self.root / 'ledger'), self.assertRaisesRegex(ValueError, 'publication changed'):
            controller.inspect_lineage(path)

    def test_lineage_rejects_missing_tampered_or_deleted_consumed_reservation(self):
        path, _, _ = self.lineage_fixture()
        ledger_path = self.root / 'ledger/seed-reservations.json'
        before = ledger_path.read_bytes()
        with patch.object(controller, 'BASE', self.root / 'ledger'):
            ledger_path.unlink()
            with self.assertRaises(FileNotFoundError):
                controller.inspect_lineage(path)
            ledger_path.write_bytes(before)
            changed = controller.read(ledger_path); changed['reservations'][0]['offset'] = 1
            controller.write(ledger_path, changed)
            with self.assertRaisesRegex(ValueError, 'ledger is inconsistent'):
                controller.inspect_lineage(path)
            controller.write(ledger_path, {'schema': 'heroes-selfplay-seeds-v1', 'reservations': []})
            with self.assertRaisesRegex(ValueError, 'no durable consumed seed reservation'):
                controller.inspect_lineage(path)

    def test_continuation_check_does_not_reserve_register_or_launch(self):
        state = SimpleNamespace(progress=SimpleNamespace(updates=1730))
        lineage = dict(admitted_games=8, remaining_games=92, selected={'checkpoint': 'published'})
        with patch.object(controller, 'activity_lease', side_effect=nullcontext), \
             patch.object(controller, 'inspect_continuation', return_value=({}, lineage, state, configuration(), {}, {})), \
             patch.object(controller, 'seed_preview', return_value={'reserved': False}), \
             patch.object(controller, 'reserve_schedule') as reserve, patch.object(controller, 'supervise') as launch, \
             patch.object(controller, 'write') as write, patch('builtins.print'):
            self.assertEqual(controller.main(['continuation-check']), 0)
        reserve.assert_not_called(); launch.assert_not_called(); write.assert_not_called()

    def test_resume_transition_rejects_changes_to_existing_helper_or_capture_logic(self):
        names = ('tools/run_selfplay_training.py', 'experiments/learner/selfplay_training.py',
                 'experiments/learner/local_script_training.py')
        actual_root, git = controller.ROOT, controller.subprocess.check_output
        before, after = {}, {}
        for name in names:
            previous = git(['git', 'show', controller.CONTINUATION_BASE + ':' + name], cwd=actual_root)
            current = (actual_root / name).read_bytes()
            before[name] = hashlib.sha256(previous).hexdigest()
            after[name] = hashlib.sha256(current).hexdigest()
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(current)
        base = dict(implementation=before, source={'source_sha256': before})
        identity = dict(implementation=after, source={'source_sha256': after})
        reviewed = {name: dict(before=before[name], after=after[name]) for name in names if before[name] != after[name]}
        def historical(args, **kwargs):
            return git(args, **{**kwargs, 'cwd': actual_root})
        with patch.object(controller, 'ROOT', self.root), patch.object(controller.subprocess, 'check_output', side_effect=historical):
            self.assertEqual(controller.validate_resume_source_transition(base, identity, reviewed), reviewed)
            helper = self.root / names[1]
            original = helper.read_text()
            self.assertIn('Self-play verification/update deadline exhausted', original)
            helper.write_text(original.replace('Self-play verification/update deadline exhausted', 'Changed verifier behavior'))
            after[names[1]] = hashlib.sha256(helper.read_bytes()).hexdigest()
            reviewed[names[1]]['after'] = after[names[1]]
            with self.assertRaisesRegex(ValueError, 'preexisting learner/admission'):
                controller.validate_resume_source_transition(base, identity, reviewed)
            helper.write_text(original)
            after[names[1]] = hashlib.sha256(helper.read_bytes()).hexdigest()
            reviewed[names[1]]['after'] = after[names[1]]
            script = self.root / names[0]
            script.write_text(script.read_text().replace("'Self-play capture failed; no retries or partial batch admission'", "'changed capture'"))
            after[names[0]] = hashlib.sha256(script.read_bytes()).hexdigest()
            reviewed[names[0]]['after'] = after[names[0]]
            with self.assertRaisesRegex(ValueError, 'capture, verification'):
                controller.validate_resume_source_transition(base, identity, reviewed)

    def test_relocated_parent_must_preserve_full_pair_and_both_file_bytes(self):
        published = dict(policy=self.artifact('published-policy.pt', {'weights': 1730}),
                         resume=self.artifact('published-resume.pt', {'adam': 33554}), progress={'updates': 1730})
        relocated = deepcopy(published)
        for name in ('policy', 'resume'):
            target = self.root / ('copied-' + name + '.pt')
            target.write_bytes(Path(published[name]['path']).read_bytes())
            relocated[name] = controller.binding(target)
        controller.check_relocated_pair(relocated, published)
        relocated['progress']['updates'] = 1731
        with self.assertRaisesRegex(ValueError, 'progress, Adam, RNG'):
            controller.check_relocated_pair(relocated, published)

    def test_original_lease_excludes_competing_launch_without_writing_locks(self):
        artifacts = self.root / 'artifacts'; original = artifacts / 'local-script-training/original'
        original.mkdir(parents=True)
        paths = [artifacts / 'local-script-training/owner.lock', original / 'owner.lock']
        for path in paths:
            path.write_text('existing owner evidence\n')
        before = [path.read_bytes() for path in paths]
        with patch.object(controller, 'ARTIFACTS', artifacts), patch.object(controller, 'ORIGINAL', original):
            with controller.activity_lease():
                with self.assertRaisesRegex(ValueError, 'owns the machine'):
                    with controller.activity_lease():
                        self.fail('Competing lease was admitted')
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_configuration_rejects_learning_and_horizon_changes(self):
        expected = configuration()
        self.assertEqual(controller.validate_configuration(expected), expected)
        for key, value in (('max_days', 30), ('max_total_choices', 10240), ('adam', {'lr': 0.001}),
                           ('allow_verified_training_prefix', True), ('ppo', {**controller.PPO, 'gamma': 0.99})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                controller.validate_configuration({**expected, key: value})

    def test_batch_phase_deadlines_preserve_parent_offsets_and_cleanup_reserve(self):
        launch = dict(configuration=configuration(), hard_deadline_monotonic=20100, hard_deadline_unix=21000)
        with patch.object(controller.time, 'monotonic', return_value=100), \
             patch.object(controller.time, 'time', return_value=1000):
            phases = controller.batch_phase_deadlines(launch)
            self.assertEqual(phases['training']['configured_seconds'], 2400)
            self.assertEqual(phases['update']['configured_seconds'], 10740)
            self.assertEqual(phases['training']['hard_deadline_monotonic'], 2500)
            self.assertEqual(phases['update']['hard_deadline_monotonic'], 10840)
            self.assertEqual(controller.phase_remaining(phases['training']), 2370)
            self.assertEqual(controller.phase_remaining(phases['update']), 10710)
            self.assertEqual(controller.phase_remaining(phases['update'], 0), 10740)
            launch.update(hard_deadline_monotonic=500, hard_deadline_unix=1400)
            capped = controller.batch_phase_deadlines(launch)
            self.assertEqual(controller.phase_remaining(capped['training']), 370)
            self.assertEqual(controller.phase_remaining(capped['update']), 370)

    def completed_collection(self, *, exhaust_phase=False):
        from experiments.headless import process_deadline
        from experiments.learner import selfplay_training as training
        from validation.probes import validate_hero_policy
        clock = [100.0]
        launch = dict(configuration=configuration(), hard_deadline_monotonic=20100, hard_deadline_unix=21000)
        rows = controller.schedule(dict(count=4, mode='diagnostic', engine_base=100, policy_base=200), 0, 4)
        plan = dict(schedule=rows, configuration=configuration(), execution_output=str(self.root / 'execution'),
                    replay_binary={'path': '/unused/replay'}, data='/unused/data')
        plan_path = self.root / 'plan.json'; controller.write(plan_path, plan, new=True)
        observed, saved, spawned = [], {}, []
        def create_binding(path, sha, index, folder):
            folder.mkdir()
            for name in ('manifest.json', 'neural-rollout.json', 'training-policy-reconstruction.json',
                         'training-verification/contextual-replay-verification.json',
                         'training-verification/native-report.json', 'training-receipt.json'):
                value = dict(status='truncated', termination_reason='max_days',
                             outcomes_by_player={'red': None, 'blue': None}, rewards_by_player={'red': 0, 'blue': 0})
                controller.write(folder / name, value, new=True)
            return index
        def spawn(command, **kwargs):
            # All captures have already completed successfully before their
            # deadline; their coordinator reads completion later in the queue.
            def communicate(**kwargs):
                clock[0] = 701
                return b'{}', None
            process = SimpleNamespace(returncode=0, communicate=communicate)
            spawned.append(kwargs['deadline_monotonic'])
            return process
        def verify(capability, learner, opponent, **kwargs):
            observed.append(kwargs['remaining']())
            if exhaust_phase:
                clock[0] = 2471  # The last 30 seconds are reserved for cleanup.
                kwargs['remaining']()
            return SimpleNamespace(collected=SimpleNamespace(rollouts=[SimpleNamespace(decisions=[]) for _ in (0, 1)]),
                                   to_dict=lambda: {'passed': True})
        with patch.object(controller.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(controller.time, 'time', side_effect=lambda: clock[0] + 900), \
             patch.object(training, 'create_capture_binding', side_effect=create_binding), \
             patch.object(training, 'register_capture_launch'), patch.object(training, 'accept_capture_completion'), \
             patch.object(training, 'verify_training_capture', side_effect=verify), \
             patch.object(process_deadline, 'guarded_popen', side_effect=spawn), \
             patch.object(process_deadline, 'close_owned_process'), \
             patch.object(process_deadline, 'cleanup_from_evidence'), \
             patch.object(validate_hero_policy, 'wall_limit', side_effect=lambda seconds: nullcontext()):
            phases = controller.batch_phase_deadlines(launch)
            result = None
            try:
                result = controller.collect_batch(plan_path, plan, object(), object(), launch,
                    lambda index, row: saved.__setitem__(index, row), phase_deadline=phases['training'])
            except TimeoutError:
                if not exhaust_phase:
                    raise
            else:
                self.assertFalse(exhaust_phase, 'An expired shared phase must not produce receipts')
        return result, observed, saved, spawned

    def test_completed_capture_can_verify_after_its_capture_deadline(self):
        result, observed, saved, spawned = self.completed_collection()
        self.assertEqual(spawned, [700] * 4)
        self.assertEqual(observed, [1769] * 4)
        self.assertEqual(len(result[0]), 4)
        self.assertTrue(all(row['status'] == 'passed' for row in saved.values()))
        self.assertEqual(saved[0]['deadlines']['capture_seconds'], 600)
        self.assertEqual(saved[0]['deadlines']['collection_and_verification']['configured_seconds'], 2400)

    def test_expired_shared_verification_phase_rejects_completed_capture(self):
        result, observed, saved, _ = self.completed_collection(exhaust_phase=True)
        self.assertIsNone(result)
        self.assertEqual(len(observed), 1)
        self.assertEqual(saved[0]['status'], 'failed')
        self.assertIn('training phase deadline exhausted', saved[0]['error'])
        self.assertTrue(controller.read(self.root / 'execution/shutdown.json')['owned_cleanup_verified'])

    def test_protection_detects_original_artifact_drift(self):
        item = self.artifact('original.json', {'checkpoint': 1728})
        self.assertTrue(controller.verify_protected({'original': item}))
        Path(item['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            controller.verify_protected({'original': item})

    def acceptance_fixture(self):
        native = self.root / 'native.json'; native.write_text('{}')
        identity = {'checkout': {'commit': 'frozen'}, 'source': {'frozen': 'source'},
                    'implementation': {'engine/source.cpp': 'engine-sha'}}
        parent_sha = 'a' * 64
        attempts = []
        for index in range(8):
            proof = {'replay_verified': True, 'histories': {'verified': True}, 'policies': [
                dict(player=player, maximum_absolute_difference=0, private_memory_reset=True, weights_unchanged=True)
                for player in (0, 1)]}
            evidence = {'training-policy-reconstruction.json': self.artifact(f'{index}/proof.json', proof),
                        'training-verification/contextual-replay-verification.json':
                            self.artifact(f'{index}/replay.json', {'verified': True})}
            attempts.append(dict(episode=index % 4 + 1, status='passed', cleanup_verified=True, replay_verified=True,
                both_policies_reconstructed=True, episode_status='truncated' if index == 0 else 'completed', evidence=evidence))
        batches = []
        for index in range(2):
            plan = dict(mode='diagnostic', source=identity['source'], implementation=identity['implementation'],
                        runtime={}, native={}, configuration=configuration(),
                        checkpoint_source={'purpose': 'diagnostic', 'initial_checkpoint': {'sha256': parent_sha}},
                        schedule=[{'episode': i + 1} for i in range(4)])
            batches.append(dict(plan=self.artifact(f'batch-{index}.json', plan), attempts=attempts[index*4:index*4+4]))
        from experiments.learner import selfplay_training as training
        checker = patch.object(training, '_plan', side_effect=lambda path, sha: controller.read(path))
        checker.start(); self.addCleanup(checker.stop)
        cutoff_checker = patch.object(controller, 'validate_cutoff_controls')
        cutoff_checker.start(); self.addCleanup(cutoff_checker.stop)
        diagnostic = dict(status='completed', mode='diagnostic', optimizer_updates=0, training_samples_admitted=0,
            owned_cleanup_verified=True, protected_inputs_unchanged=True, source=identity,
            parent_policy_sha256=parent_sha, native_manifest=controller.binding(native), policy_support=controller.GUARD,
            configuration=configuration(), batches=batches, runtime={}, native={})
        copied = dict(status='passed', diagnostic_optimizer_updates=1, official_updates=0,
            exact_model_adam_rng_progress_pool=True, source=identity,
            parent_policy_sha256=parent_sha, native_manifest=controller.binding(native))
        gate = dict(schema='heroes-selfplay-acceptance-v1', accepted_for_training=True, source=identity,
            parent_policy_sha256=parent_sha, native_manifest=controller.binding(native),
            diagnostic_report=self.artifact('diagnostic.json', diagnostic),
            copied_update_report=self.artifact('copied.json', copied),
            cutoff_report=self.artifact('cutoff-summary.json', {'separately_tested': True}),
            offline_tests=self.artifact('tests.json', {'status': 'passed', 'source': identity}),
            prepare_vcmi_check=self.artifact('engine.json', {'status': 'passed', 'engine_inventory': identity['implementation']}))
        path = self.artifact('gate.json', gate)['path']
        pointer = {'parent': {'policy': {'sha256': parent_sha},
                             'plan': self.artifact('parent-plan.json', {'runtime': {}, 'native': {}})}}
        return native, identity, pointer, Path(path), gate, diagnostic

    def test_acceptance_requires_eight_persisted_both_policy_proofs(self):
        native, identity, pointer, path, gate, diagnostic = self.acceptance_fixture()
        with patch.object(controller, 'NATIVE', native):
            self.assertEqual(controller.validate_acceptance(path, identity, pointer), controller.binding(path))
            first = diagnostic['batches'][0]['attempts'][0]
            first['both_policies_reconstructed'] = False
            controller.write(self.root / 'diagnostic.json', diagnostic)
            gate['diagnostic_report'] = controller.binding(self.root / 'diagnostic.json')
            controller.write(path, gate)
            with self.assertRaisesRegex(ValueError, 'Unverified diagnostic'):
                controller.validate_acceptance(path, identity, pointer)

    def test_acceptance_rejects_stale_copied_update_source(self):
        native, identity, pointer, path, gate, _ = self.acceptance_fixture()
        copied = controller.read(self.root / 'copied.json'); copied['source'] = {'old': True}
        controller.write(self.root / 'copied.json', copied)
        gate['copied_update_report'] = controller.binding(self.root / 'copied.json'); controller.write(path, gate)
        with patch.object(controller, 'NATIVE', native), self.assertRaisesRegex(ValueError, 'copied optimizer'):
            controller.validate_acceptance(path, identity, pointer)

    def test_restore_failure_finishes_a_stopped_report_without_update(self):
        from experiments.learner import local_script_training as local, selfplay_training as training
        output = self.root / 'run'; output.mkdir()
        launch = dict(output=str(output), mode='diagnostic', source={}, parent_policy_sha256='a' * 64,
            native_manifest={}, runtime={}, native={}, configuration=configuration(), protected={}, supervisor_pid=os.getppid(),
            execution={}, parent_descriptor={}, reservation={'count': 8})
        args = SimpleNamespace(plan=self.root / 'launch.json', sha256='b' * 64)
        with patch.object(controller, 'load_launch', return_value=launch), \
             patch.object(controller, 'configure_native'), \
             patch.object(local, 'restore_parent', side_effect=ValueError('parent drift')), \
             patch.object(training, 'native_update') as update:
            result = controller.coordinator(args)
        self.assertEqual(result['status'], 'stopped')
        self.assertIn('parent drift', result['error'])
        self.assertEqual(result['optimizer_updates'], 0)
        update.assert_not_called()
        self.assertEqual(controller.read(output / 'report.json')['status'], 'stopped')

    def test_diagnostic_runs_two_four_game_batches_and_never_updates(self):
        from experiments.learner import local_script_training as local, selfplay_training as training
        output = self.root / 'run'; output.mkdir()
        launch = dict(output=str(output), mode='diagnostic', source={}, parent_policy_sha256='a' * 64,
            native_manifest={}, runtime={}, native={}, configuration=configuration(), protected={}, supervisor_pid=os.getppid(),
            execution={}, parent_descriptor={}, original_parent={'policy': {}}, reservation=dict(count=8,
                mode='diagnostic', engine_base=100, policy_base=200),
            hard_deadline_monotonic=time.monotonic() + 300, hard_deadline_unix=time.time() + 300)
        state = SimpleNamespace(model=object())
        observed = []
        def plan(launch, state, index, start, count, previous):
            observed.append((start, count))
            folder = output / f'batch-{index:03d}'; folder.mkdir()
            value = dict(schedule=controller.schedule(launch['reservation'], start, count), execution_output=str(folder / 'execution'))
            controller.write(folder / 'plan.json', value, new=True)
            return folder / 'plan.json', value, object()
        def collect(path, plan, learner, opponent, launch, save, *, phase_deadline):
            rows = [dict(**row, status='passed') for row in plan['schedule']]
            controller.write(Path(plan['execution_output']) / 'shutdown.json', {'owned_cleanup_verified': True}, new=True)
            for index, row in enumerate(rows):
                save(index, row)
            return (object(),) * 4, rows
        with patch.object(controller, 'load_launch', return_value=launch), patch.object(controller, 'configure_native'), \
             patch.object(local, 'restore_parent', return_value=state), patch.object(controller, 'batch_plan', side_effect=plan), \
             patch.object(controller, 'collect_batch', side_effect=collect), patch.object(training, 'release_receipts'), \
             patch.object(training, 'native_update') as update:
            result = controller.coordinator(SimpleNamespace(plan=self.root / 'launch.json', sha256='b' * 64))
        self.assertEqual(observed, [(0, 4), (4, 4)])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertEqual(result['training_samples_admitted'], 0)
        self.assertFalse((output / 'latest-checkpoint.json').exists())
        update.assert_not_called()

    def complete_update(self, *, after_pair_error=None, admitted_before=0):
        import torch
        from experiments.learner import local_script_training as local, selfplay_training as training
        from experiments.learner.builtin_training_resume import ParentState
        from experiments.learner.checkpoints import TrainingProgress
        @dataclass
        class Result:
            samples: int = 12
            optimizer_steps: int = 1
        output = self.root / 'run'; output.mkdir()
        model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
        state = ParentState(model, object(), object(), object(), TrainingProgress(1728 + admitted_before // 4, 6912 + admitted_before, 978803),
                            {'old_sampler': 1}, {}, 'parent', 'old', {}, 500)
        launch = dict(output=str(output), mode='training', source={}, parent_policy_sha256='a' * 64,
            native_manifest={}, runtime={}, native={}, configuration=configuration(), protected={}, supervisor_pid=os.getppid(),
            execution={}, parent_descriptor={}, original_parent={'policy': {}}, campaign_id='test',
            reservation=dict(count=4, mode='training', engine_base=100, policy_base=200),
            hard_deadline_monotonic=time.monotonic() + 300, hard_deadline_unix=time.time() + 300)
        if admitted_before:
            launch.update(admitted_before=admitted_before, resume_parent={}, parent_optimizer_steps=500,
                          resume_pair={'metadata': {}, 'policy': {}, 'sampler_state': {'optimizer_steps': 500}})
        folder = output / 'batch-000'; folder.mkdir()
        plan = dict(schedule=controller.schedule(launch['reservation'], 0, 4), opponent_checkpoint={},
                    execution_output=str(folder / 'execution'))
        controller.write(folder / 'plan.json', plan, new=True)
        def collect(path, plan, learner, opponent, launch, save, *, phase_deadline):
            rows = [dict(**row, status='passed') for row in plan['schedule']]
            controller.write(Path(plan['execution_output']) / 'shutdown.json', {'owned_cleanup_verified': True}, new=True)
            for index, row in enumerate(rows):
                save(index, row)
            return (object(),) * 4, rows
        def pair(output, model, optimizer, **kwargs):
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            controller.write(output / 'candidate.json', {'verified_pair': True}, new=True)
            return dict(exact_model_adam_rng_progress_pool=True, policy={'path': 'private-policy'},
                        weights_sha256='new', metadata=kwargs['metadata'], progress=vars(kwargs['progress']),
                        sampler_state=kwargs['sampler_state'])
        checks = [launch, after_pair_error or launch, launch]
        with patch.object(controller, 'load_launch', side_effect=checks), patch.object(controller, 'configure_native'), \
             patch.object(local, 'restore_parent', return_value=state), \
             patch.object(training, 'restore_checkpoint_pair', return_value=state), \
             patch.object(controller, 'batch_plan', return_value=(folder / 'plan.json', plan, object())), \
             patch.object(controller, 'collect_batch', side_effect=collect), \
             patch.object(training, 'native_update', return_value=(Result(), (object(),) * 4)), \
             patch.object(training, 'metadata', return_value={'purpose': 'training'}), \
             patch.object(training, 'save_checkpoint_pair', side_effect=pair):
            result = controller.coordinator(SimpleNamespace(plan=self.root / 'launch.json', sha256='b' * 64))
        return result, output, model, state

    def test_complete_update_keeps_adam_parameters_trainable_until_pair_verifies(self):
        result, output, model, state = self.complete_update()
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['admitted_games'], 4)
        self.assertEqual(result['cumulative_progress'], {'updates': 1729, 'rollouts': 6916, 'decisions': 978815})
        pointer = controller.read(output / 'latest-checkpoint.json')
        self.assertEqual(pointer['purpose'], 'training')
        self.assertEqual(pointer['admitted_games'], 4)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(state.progress.updates, 1728)  # Original frozen ParentState was replaced, not mutated.

    def test_resumed_update_preserves_global_sampler_and_cumulative_admission(self):
        result, output, _, state = self.complete_update(admitted_before=8)
        self.assertEqual(result['status'], 'completed', result.get('error'))
        self.assertEqual(result['admitted_games'], 4)
        pointer = controller.read(output / 'latest-checkpoint.json')
        self.assertEqual(pointer['admitted_games'], 12)
        self.assertEqual(pointer['window_admitted_games'], 4)
        self.assertEqual(pointer['checkpoint']['progress']['updates'], 1731)
        sampler = pointer['checkpoint']['sampler_state']
        self.assertEqual((sampler['next_batch_index'], sampler['consumed_schedule_offset']), (3, 12))
        self.assertEqual(sampler['original_parent_sampler'], {'old_sampler': 1})
        self.assertEqual(state.progress.updates, 1730)

    def test_source_drift_after_candidate_save_prevents_admission_and_latest_publication(self):
        result, output, _, _ = self.complete_update(after_pair_error=ValueError('source changed during PPO'))
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(result['admitted_games'], 0)
        self.assertEqual(result['optimizer_updates'], 0)
        self.assertIn('source changed during PPO', result['error'])
        self.assertTrue((output / 'batch-000/checkpoint/candidate.json').exists())
        self.assertFalse((output / 'latest-checkpoint.json').exists())

    def test_all_worker_ownership_is_closed_before_any_native_cleanup_check(self):
        from experiments.headless import process_deadline
        events = []
        output = self.root / 'execution'; output.mkdir()
        (output / 'native-000001-deadline.jsonl').touch()
        jobs = [dict(index=i, process=i, log=SimpleNamespace(close=lambda: None)) for i in range(4)]
        def close(process, **kwargs):
            events.append(('worker', process))
            if process == 0:
                raise TimeoutError('first worker cleanup timed out')
        def native(path, **kwargs):
            self.assertEqual(events, [('worker', i) for i in range(4)])
            events.append(('native', path.name))
        with patch.object(process_deadline, 'close_owned_process', side_effect=close), \
             patch.object(process_deadline, 'cleanup_from_evidence', side_effect=native):
            errors = controller.close_collection_jobs(jobs, output,
                {'hard_deadline_monotonic': 1, 'hard_deadline_unix': 1})
        self.assertEqual(len(errors), 1)
        self.assertIn('worker 0', errors[0])
        self.assertEqual(events[-1][0], 'native')

    def test_cutoff_acceptance_requires_both_cases_both_seats_and_actual_policy_answers(self):
        from tools import validate_selfplay_cutoffs as controls
        identity = {'checkout': {'commit': 'frozen'}, 'source': {'weights': 'source'}, 'implementation': {'core': 'sha'}}
        cutoff_source = deepcopy(identity); cutoff_source['implementation']['additional-probe'] = 'bound'
        parent = {'policy': {'sha256': 'a' * 64}}
        rows = controls.schedule()
        plan = dict(source=cutoff_source, native_runtime={}, policy_support=controller.GUARD,
                    full_model_adam_rng_progress_sampler_pool_restored=True, parent_descriptor=parent, schedule=rows)
        plan_binding = self.artifact('cutoff/plan.json', plan)
        attempts = []
        for index, row in enumerate(rows):
            attempt = dict(**row, status='passed', cleanup_verified=True, history_audit_verified=True,
                           replay_verified=True, episode_status='truncated', training_input=False,
                           evidence={'replay': self.artifact(f'cutoff/{index}/replay.json', {'verified': True})})
            if row['case'] == 'mandatory-level-up':
                attempt['mixed_policy_reconstruction'] = dict(passed=True, maximum_absolute_difference=0,
                    private_memory_reset=True, weights_unchanged=True, training_input=False, actions_resampled=1)
            attempts.append(attempt)
        report = dict(schema=controls.SCHEMA, status='passed', not_started=0, training_input=False,
            training_samples=0, optimizer_updates=0, owned_cleanup_verified=True, protected_inputs_unchanged=True,
            plan=plan_binding, source=cutoff_source, native_runtime={}, parent_descriptor=parent, attempts=attempts)
        report_binding = self.artifact('cutoff/report.json', report)
        self.artifact('cutoff/shutdown.json', {'owned_cleanup_verified': True, 'returncode': 0,
                                             'protected_inputs_unchanged': True})
        with patch.object(controls, 'check_identity'):
            controller.validate_cutoff_controls(report_binding, identity, {'parent': parent}, {'runtime': {}})
            report['attempts'][-1]['mixed_policy_reconstruction']['actions_resampled'] = 0
            controller.write(report_binding['path'], report)
            with self.assertRaisesRegex(ValueError, 'Mandatory cutoff response'):
                controller.validate_cutoff_controls(controller.binding(report_binding['path']), identity,
                                                     {'parent': parent}, {'runtime': {}})


if __name__ == '__main__':
    unittest.main()
