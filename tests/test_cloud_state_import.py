"""Offline cloud import admission, consumed-seed and subsequent Mac lineage guards."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.learner import local_script_training as adapter
from tools import cloud_state_import as cloud
from tools import run_local_training as runner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def complete_report():
    return dict(status='finished', worker_complete=True, supervisor_cleanup_verified=True,
                within_shared_budget=True, optimizer_updates=1, update_status='checkpoint_verified',
                attempts=[dict(status='passed', completion_receipt=True, cleanup_verified=True,
                               identity_verified=True, replay_verified=True) for _ in range(4)])


def batch_schedule(sequence):
    return [dict(engine_seed=40_000_000+4*sequence+i, policy_seed=50_000_000+4*sequence+i,
                 learner_player=sequence % 2) for i in range(4)]


class ImportFixture(unittest.TestCase):
    """Tiny synthetic publications exercise metadata gates without restoring a model."""
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.runs = self.root / 'runs'; self.runs.mkdir()
        self.anchor = self.runs / 'original699'
        self.config = dict(run_root=str(self.runs), default_profile='original',
                           seed_namespace={'engine_base': 40_000_000, 'policy_base': 50_000_000})
        self.remote = self.root / 'private-cloud'
        self.marker = self.runs / 'cloud-import.json'
        self.base_ledger = dict(next_sequence=869, consumed=[dict(sequence=336), dict(sequence=868)])
        self.base_parent = self.make_descriptor(self.root / 'base1229')
        self.base = dict(admitted_before_session=2400, next_sequence=869,
                         seed_ledger_snapshot=self.base_ledger, evidence={})
        self.base_receipt = self.root / 'recovery1229.json'
        write(self.base_receipt, dict(verification=dict(original_moving_snapshots={
            'latest-run.json': '{}', 'latest-checkpoint.json': '{}',
            'sequence-ledger.json': json.dumps(self.base_ledger)})))
        self.session = dict(source=dict(git_commit=cloud.CLOUD_COMMIT, tracked_worktree_dirty=False),
            profile='original', scope=adapter.SCOPE, policy_support='visible-interaction-budget-v1',
            training_engine_seed_base=40_000_000, training_policy_seed_base=50_000_000,
            initial_parent=self.base_parent, original_continuation=dict(origin_run=str(self.anchor)),
            configuration={'fixed': True}, execution={'fixed': True}, target_games=2600, next_sequence=869)
        write(self.remote / 'closure/session.json', self.session)
        session_binding = cloud.binding(self.remote / 'closure/session.json')
        previous = self.base_parent
        self.rows = []
        self.candidates = {}
        self.manifests = {}
        for sequence in range(869, 975):
            folder = self.remote / f'batches/train-{sequence:06d}-seat{sequence%2}'
            plan = dict(source=dict(git_commit=cloud.CLOUD_COMMIT), profile='original', mode='training',
                scope=adapter.SCOPE, batch_sequence=sequence, parent_descriptor=previous,
                configuration=self.session['configuration'], execution=self.session['execution'],
                policy_support=self.session['policy_support'], local_session=session_binding,
                map=dict(sha256=cloud.MAP_SHA256, scenario='hero-management-v2', profile='original', fixture=False),
                native_opponents=[dict(module='Nullkiller', difficulty=0, kind='builtin', player=i) for i in (0, 1)],
                schedule=batch_schedule(sequence))
            progress = dict(updates=sequence+361, rollouts=4*(sequence+361), decisions=sequence*10)
            report = {**complete_report(), 'cumulative_progress': progress}
            actual = self.make_descriptor(folder, plan, report)
            historical = {key: {**value, 'path': '/workspace/cloud/' + str(sequence) + '/' + key}
                          for key, value in actual.items()}
            candidate = dict(parent=historical, progress=progress)
            write(folder / 'candidate.json', candidate)
            self.candidates[sequence] = candidate
            manifest = self.make_manifest(folder)
            self.manifests[str(folder.relative_to(self.remote)) + '/manifest.json'] = manifest
            row = dict(mode='training', sequence=sequence, parent=previous,
                       plan=historical['plan'], report=historical['report'], status='passed', checkpoint_verified=True,
                       private_publication=dict(ref=cloud.RESULTS_REF,
                           prefix='cloud/' + cloud.HANDOFF + '/' + str(folder.relative_to(self.remote)),
                           manifest_sha256=manifest['sha256'], fresh_readback_verified=True,
                           checkpoint_roundtrip_verified=True))
            if sequence == 974:
                row.update(status='failed', checkpoint_verified=False, owned_cleanup_verified=True,
                           completed_games=4, error='TimeoutExpired git show HEAD:raw.part00000 after 120 seconds')
            self.rows.append(row)
            previous = historical
        selected = self.candidates[973]
        self.cloud_ledger = dict(next_sequence=975, consumed=self.base_ledger['consumed'] +
                                [dict(sequence=n) for n in range(869, 975)])
        self.campaign = dict(status='stopped_failure', initial_state_unchanged=True, owned_cleanup_verified=True,
            full_final_restore=dict(exact=True, progress=selected['progress']), latest_progress=selected['progress'],
            latest_parent=selected['parent'], batches=self.rows)
        for name, value in {'campaign-report.json': self.campaign,
                'shutdown.json': dict(owned_cleanup_verified=True, worker_returncode=1, within_hard_deadline=True),
                'latest-checkpoint.json': selected, 'pending-checkpoint.json': self.candidates[974],
                'sequence-ledger.json': self.cloud_ledger}.items():
            write(self.remote / 'closure' / name, value)
        write(self.remote / 'start/session.json', self.session)
        write(self.remote / 'start/sequence-ledger.json', self.base_ledger)
        for folder in ('closure', 'start'):
            self.manifests[folder + '/manifest.json'] = self.make_manifest(self.remote / folder)
        self.closure = self.root / 'process-closure.json'
        write(self.closure, dict(schema='heroes-cloud1335-process-closure-v1', handoff_id=cloud.HANDOFF,
            closure_commit=cloud.CLOSURE_COMMIT, method='live_process_snapshot', live_owned_processes=[], supervisor_absent=True,
            automatic_restart=False, cloud_launch_fenced=True, training_launched=False, observed_utc='test-only'))
        self.state = dict(full_state_sha256='a'*64, component_sha256={}, optimizer_steps=1335)
        self.recovery = self.root / 'publication-recovery.json'
        write(self.recovery, dict(schema='heroes-cloud1335-publication-recovery-v1', handoff_id=cloud.HANDOFF,
            closure_commit=cloud.CLOSURE_COMMIT, sequence=974, progress=self.candidates[974]['progress'],
            fresh_readback_verified=True, raw_inventory_verified=True, checkpoint_roundtrip_verified=True,
            original_failure_preserved=True, update_reapplied=False, games_replayed=0, training_launched=False,
            manifest_sha256=self.manifests['batches/train-000974-seat0/manifest.json']['sha256'],
            full_state_sha256=self.state['full_state_sha256'], commit='b'*40, ref='refs/heads/codex/test-recovery'))
        self.receipt = dict(schema=cloud.SCHEMA, handoff_id=cloud.HANDOFF, cloud_source_commit=cloud.CLOUD_COMMIT,
            closure_commit=cloud.CLOSURE_COMMIT, run_root=str(self.runs), evidence_root=str(self.remote),
            manifests=self.manifests, base_recovery_receipt=cloud.binding(self.base_receipt),
            cloud_closure=cloud.binding(self.closure), publication_recovery=cloud.binding(self.recovery),
            selected_update=1335, state=self.state)
        self.refresh()

    def make_descriptor(self, folder, plan=None, report=None):
        folder.mkdir(parents=True)
        write(folder / 'plan.json', plan or {})
        write(folder / 'report.json', report or {})
        for name in ('after-policy.pt', 'after-resume.pt', 'initial-checkpoint.pt'):
            (folder / name).write_bytes(name.encode())
        return cloud.descriptor(folder)

    def make_manifest(self, folder):
        entries = {p.name: {key: cloud.binding(p)[key] for key in ('bytes', 'sha256')}
                   for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}
        write(folder / 'manifest.json', dict(schema='heroes-linux-private-publication-v1',
            game_data_symlinks_followed=False, entries=entries, raw_inventory={}))
        return cloud.binding(folder / 'manifest.json')

    def refresh(self):
        write(self.marker, self.receipt)
        self.profile = dict(receipt=cloud.binding(self.marker))

    def audit(self, path=None, preview=False):
        with patch.object(cloud, 'reviewed_profile', return_value=self.profile), \
                patch.object(adapter, 'read_descriptor', side_effect=lambda d:
                             (cloud.read(d['plan']['path']), cloud.read(d['report']['path']))), \
                patch.object(cloud, 'verify_state', return_value=self.candidates[
                    self.receipt['selected_update']-361]['progress']) as restore:
            result = cloud.audit_import(path or self.marker, self.config, self.anchor,
                local_auditor=lambda *a, **kw: (self.base_parent, 2600, self.base), preview=preview)
        self.assertEqual(restore.call_count, 1)
        return result

    def test_complete_pending_update_requires_certificate_and_is_selected_once(self):
        audit = self.audit()
        self.assertEqual(audit['selected_update'], 1335)
        self.assertEqual(audit['ledger']['next_sequence'], 975)
        self.assertEqual(audit['progress']['updates'], 1335)
        self.assertEqual(cloud.read(self.remote / 'closure/latest-checkpoint.json')['progress']['updates'], 1334)
        for field, value in [('raw_inventory_verified', False), ('fresh_readback_verified', False),
                             ('update_reapplied', True), ('games_replayed', 4)]:
            with self.subTest(field=field):
                original = cloud.read(self.recovery)
                write(self.recovery, {**original, field: value})
                self.receipt['publication_recovery'] = cloud.binding(self.recovery); self.refresh()
                with self.assertRaisesRegex(ValueError, 'lacks exact private publication recovery'):
                    self.audit()
                write(self.recovery, original)
                self.receipt['publication_recovery'] = cloud.binding(self.recovery); self.refresh()

    def test_unrecovered1334_selection_still_consumes_failed974(self):
        self.receipt['selected_update'] = 1334
        self.receipt.pop('publication_recovery')
        self.refresh()
        audit = self.audit()
        self.assertEqual(audit['progress']['updates'], 1334)
        self.assertEqual(audit['ledger']['consumed'][-1]['sequence'], 974)
        self.assertEqual(audit['ledger']['next_sequence'], 975)

    def test_preview_is_read_only_and_keeps_official_destination(self):
        candidate = self.root / 'private-candidate.json'
        candidate.write_bytes(self.marker.read_bytes())
        self.marker.unlink()
        audit = self.audit(candidate, preview=True)
        self.assertEqual(audit['import_run'], str(self.runs / 'cloud-import-1335'))
        self.assertEqual(audit['receipt']['path'], str(self.marker))
        self.assertFalse(self.marker.exists())
        with self.assertRaisesRegex(ValueError, 'official selection'):
            self.audit(candidate)

    def test_changed_profile_marker_or_evidence_is_rejected(self):
        self.marker.write_text(self.marker.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'official selection'):
            self.audit()
        self.refresh()
        (self.remote / 'batches/train-000900-seat0/after-resume.pt').write_bytes(b'changed Adam')
        with self.assertRaisesRegex(ValueError, 'Published metadata/state changed'):
            self.audit()

    def test_missing_batch_or_unclean_cloud_closure_is_rejected(self):
        removed = self.receipt['manifests'].pop('batches/train-000900-seat0/manifest.json')
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'omitted or added'):
            self.audit()
        self.receipt['manifests']['batches/train-000900-seat0/manifest.json'] = removed
        write(self.closure, {**cloud.read(self.closure), 'live_owned_processes': [1234]})
        self.receipt['cloud_closure'] = cloud.binding(self.closure); self.refresh()
        with self.assertRaisesRegex(ValueError, 'closed and fenced'):
            self.audit()

    def test_manifest_rejects_traversal_and_symlink_members(self):
        folder = self.remote / 'start'
        manifest = cloud.read(folder / 'manifest.json')
        manifest['entries']['../escape'] = dict(bytes=0, sha256='0'*64)
        write(folder / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Unsafe publication member'):
            cloud.manifest_files(self.remote, 'start', cloud.binding(folder / 'manifest.json'))
        manifest['entries'].pop('../escape')
        target = self.root / 'outside'; target.write_bytes(b'outside')
        (folder / 'link').symlink_to(target)
        manifest['entries']['link'] = {key: cloud.binding(target)[key] for key in ('bytes', 'sha256')}
        write(folder / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'absolute regular file'):
            cloud.manifest_files(self.remote, 'start', cloud.binding(folder / 'manifest.json'))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.anchor = self.root / 'original699'
        self.config = dict(run_root=str(self.root),
                           seed_namespace={'engine_base': 40_000_000, 'policy_base': 50_000_000})
        self.marker = self.root / 'cloud-import.json'; write(self.marker, {})
        self.audit = dict(receipt=cloud.binding(self.marker), parent={'plan': {'sha256': 'cloud1335'}},
            progress=dict(updates=1335, rollouts=5340, decisions=680677), selected_update=1335,
            ledger=dict(next_sequence=975, consumed=[dict(sequence=336), dict(sequence=974)]),
            base={'evidence': {}}, import_run=str(self.root / 'cloud-import-1335'))
        self.initial = cloud.moving_pointers(self.audit)
        for name, value in self.initial.items():
            write(self.root / name, value)

    def select(self):
        with patch.object(cloud, 'audit_import', return_value=self.audit):
            return cloud.continuation(self.anchor, self.config, self.marker, local_auditor=None)

    def descendant(self, name='next', sequence=975, parent=None, progress=None, ledger=None):
        run = self.root / name
        run.mkdir()
        proof = dict(origin_run=str(self.anchor), evidence={cloud.RECEIPT_KEY: self.audit['receipt']},
                     seed_ledger_snapshot=deepcopy(ledger or self.audit['ledger']))
        session = dict(run_root=str(self.root), run_dir=str(run), profile='original', original_continuation=proof,
                       training_engine_seed_base=40_000_000, training_policy_seed_base=50_000_000)
        write(run / 'session.json', session)
        child = {'plan': {'sha256': name}}
        progress = progress or dict(updates=1336, rollouts=5344, decisions=680700)
        report = dict(status='stopped_budget', initial_state_unchanged=True, owned_cleanup_verified=True,
            full_final_restore=dict(exact=True, progress=progress), latest_progress=progress,
            latest_parent=child, batches=[dict(mode='training', sequence=sequence)])
        write(run / 'campaign-report.json', report)
        write(run / 'shutdown.json', dict(owned_cleanup_verified=True, within_hard_deadline=True, worker_returncode=0))
        write(self.root / 'latest-run.json', dict(run_dir=str(run), session=cloud.binding(run / 'session.json')))
        write(self.root / 'latest-checkpoint.json', dict(run_dir=str(run), parent=child, progress=progress))
        current = deepcopy(ledger or self.audit['ledger'])
        current['consumed'].append(dict(sequence=sequence)); current['next_sequence'] = sequence + 1
        write(self.root / 'sequence-ledger.json', current)
        plan = dict(profile='original', mode='training', scope=adapter.SCOPE, batch_sequence=sequence,
            local_session=cloud.binding(run / 'session.json'), schedule=batch_schedule(sequence),
            parent_descriptor=parent or self.audit['parent'])
        return run, child, plan, complete_report()

    def test_initial_import_accounts_recovered_games_without_replaying975(self):
        parent, remaining, proof = self.select()
        self.assertEqual(parent, self.audit['parent'])
        self.assertEqual((remaining, proof['admitted_before_session'], proof['next_sequence']), (2176, 2824, 975))
        self.assertEqual(proof['evidence'][cloud.RECEIPT_KEY], self.audit['receipt'])

    def test_complete_mac_update_reduces_original_target_once(self):
        _, child, plan, report = self.descendant()
        with patch.object(adapter, 'read_descriptor', return_value=(plan, report)):
            parent, remaining, proof = self.select()
        self.assertEqual((parent, remaining), (child, 2172))
        self.assertEqual(proof['next_sequence'], 976)

    def test_consumed_cloud_ledger_row_cannot_be_dropped_or_changed(self):
        value = deepcopy(self.audit['ledger'])
        value['consumed'][0]['reserved_utc'] = 'changed'
        write(self.root / 'sequence-ledger.json', value)
        with self.assertRaisesRegex(ValueError, 'dropped or changed'):
            self.select()

    def test_original_seed_namespace_cannot_change(self):
        self.config['seed_namespace']['engine_base'] = 20_000_000
        with self.assertRaisesRegex(ValueError, 'seed namespace changed'):
            self.select()

    def test_latest_pointer_cannot_roll_back_after_newer_mac_update(self):
        self.descendant()
        write(self.root / 'latest-checkpoint.json', self.initial['latest-checkpoint.json'])
        with self.assertRaisesRegex(ValueError, 'cannot discard'):
            self.select()

    def test_import_only_newest_marker_cannot_hide_later_update(self):
        self.descendant()
        write(self.root / 'latest-run.json', self.initial['latest-run.json'])
        with self.assertRaisesRegex(ValueError, 'cannot hide'):
            self.select()

    def test_failed_attempt_and_ancestor_snapshot_reservations_are_retained(self):
        snapshot = deepcopy(self.audit['ledger'])
        snapshot['consumed'].append(dict(sequence=975, reserved_utc='failed attempt'))
        snapshot['next_sequence'] = 976
        _, _, plan, report = self.descendant(sequence=976, ledger=snapshot)
        with patch.object(adapter, 'read_descriptor', return_value=(plan, report)):
            self.assertEqual(self.select()[2]['next_sequence'], 977)
            current = cloud.read(self.root / 'sequence-ledger.json')
            current['consumed'] = [row for row in current['consumed'] if row['sequence'] != 975]
            write(self.root / 'sequence-ledger.json', current)
            with self.assertRaisesRegex(ValueError, 'ancestor reservation snapshot'):
                self.select()

    def test_newest_failed_before_update_must_be_closed_and_keep_latest_state(self):
        run, _, _, _ = self.descendant()
        report = cloud.read(run / 'campaign-report.json')
        report.update(latest_parent=self.audit['parent'], latest_progress=self.audit['progress'],
                      full_final_restore=dict(exact=True, progress=self.audit['progress']))
        write(run / 'campaign-report.json', report)
        write(self.root / 'latest-checkpoint.json', self.initial['latest-checkpoint.json'])
        self.assertEqual(self.select()[1], 2176)
        write(run / 'shutdown.json', dict(owned_cleanup_verified=False, within_hard_deadline=True, worker_returncode=1))
        with self.assertRaisesRegex(ValueError, 'verifiably closed'):
            self.select()

    def test_missing_receipt_boundary_or_replayed_cloud_seed_is_rejected(self):
        run, _, plan, report = self.descendant()
        with patch.object(adapter, 'read_descriptor', return_value=(plan, report)):
            plan['schedule'][0]['engine_seed'] -= 4
            with self.assertRaisesRegex(ValueError, 'schedule changed or reused'):
                self.select()
            plan['schedule'] = batch_schedule(975)
            session = cloud.read(run / 'session.json')
            session['original_continuation']['evidence'] = {}
            write(run / 'session.json', session)
            write(self.root / 'latest-run.json', dict(run_dir=str(run), session=cloud.binding(run / 'session.json')))
            with self.assertRaisesRegex(ValueError, 'cloud import boundary'):
                self.select()

    def test_runner_routes_marker_and_preserves_other_unimported_roots(self):
        sentinel = ({'parent': True}, 2176, {'proof': True})
        with patch.object(cloud, 'continuation', return_value=sentinel) as imported, \
                patch.object(cloud, 'import_required', return_value=False), \
                patch.object(runner, '_local_original_continuation') as local:
            self.assertEqual(runner.original_continuation(self.anchor, self.config), sentinel)
            imported.assert_called_once()
            local.assert_not_called()
            self.marker.unlink()
            runner.original_continuation(self.anchor, self.config)
            local.assert_called_once_with(self.anchor, self.config, None)

    def test_reviewed_import_cannot_fall_back_when_marker_and_pointers_are_reset(self):
        profile_path = self.root / 'reviewed-profile.json'; write(profile_path, {})
        profile = dict(receipt=cloud.binding(self.marker))
        self.marker.unlink()
        for name in ('latest-run.json', 'latest-checkpoint.json', 'sequence-ledger.json'):
            write(self.root / name, {'restored_old1229': True})
        with patch.object(cloud, 'PROFILE', profile_path), \
                patch.object(cloud, 'reviewed_profile', return_value=profile), \
                patch.object(runner, '_local_original_continuation') as local:
            with self.assertRaisesRegex(ValueError, 'absolute regular file'):
                runner.original_continuation(self.anchor, self.config)
        local.assert_not_called()


class DeletedEnvironmentClosureTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.folder = self.root / 'closure'
        self.session = dict(source=dict(git_commit=cloud.CLOUD_COMMIT, tracked_worktree_dirty=False),
            started_unix=1793275142.1684587, absolute_deadline_unix=1793311142.1684587, supervisor_pid=95785,
            linux_cloud=dict(handoff_id=cloud.HANDOFF,
                claim=dict(ref=cloud.CLAIM_REF, commit=cloud.CLAIM_COMMIT), results_ref=cloud.RESULTS_REF))
        write(self.folder / 'session.json', self.session)
        write(self.folder / 'shutdown.json', {'checked_separately_by_import_audit': True})
        write(self.folder / 'campaign-report.json', {'checked_separately_by_import_audit': True})
        files = dict(app_unavailable=dict(schema='heroes-cloud-runtime-access-observation-v1',
            thread_id=cloud.CLOUD_THREAD, host_id='durable', is_error=True, error=cloud.DELETED_BUILD_ERROR,
            live_process_snapshot_obtained=False, training_launched=False),
            remote_claim_ref=dict(ref=cloud.CLAIM_REF, object=dict(sha=cloud.CLAIM_COMMIT, type='commit')),
            remote_results_ref=dict(ref=cloud.RESULTS_REF, object=dict(sha=cloud.CLOSURE_COMMIT, type='commit')),
            remote_claim=dict(handoff_id=cloud.HANDOFF, pid=95785, process_group=95785,
                started_unix=self.session['started_unix'], hard_deadline_unix=self.session['absolute_deadline_unix'],
                source=self.session['source']))
        self.proof = dict(schema='heroes-cloud1335-process-closure-v1', handoff_id=cloud.HANDOFF,
            closure_commit=cloud.CLOSURE_COMMIT, observed_utc='test-only', automatic_restart=False,
            method='published_terminal_training_closure_deleted_environment', training_launched=False,
            live_process_snapshot_obtained=False, outer_supervisor_absence_verified=False,
            cloud_lock_mode_changed=False, training_worker_closed=True, cloud_launch_claim_consumed=True,
            cloud_training_excluded=True)
        for name, value in files.items():
            path = self.root / (name + '.json'); write(path, value)
            self.proof[name] = cloud.binding(path)
        for key, name in (('cloud_session', 'session.json'), ('cloud_shutdown', 'shutdown.json'),
                          ('cloud_campaign', 'campaign-report.json')):
            self.proof[key] = cloud.binding(self.folder / name)
        source = self.root / 'frozen-supervisor.py'; source.write_text('# Synthetic closure proof fixture.\n')
        self.proof['frozen_supervisor'] = cloud.binding(source)

    def check(self):
        with patch.object(cloud, 'FROZEN_SUPERVISOR_SHA256', self.proof['frozen_supervisor']['sha256']):
            cloud.check_cloud_closure(self.proof, self.folder)

    def test_exact_terminal_evidence_does_not_claim_current_process_inventory(self):
        self.check()
        for key, value in [('live_process_snapshot_obtained', True), ('outer_supervisor_absence_verified', True),
                           ('cloud_lock_mode_changed', True), ('supervisor_absent', True)]:
            with self.subTest(key=key):
                original = deepcopy(self.proof)
                self.proof[key] = value
                with self.assertRaisesRegex(ValueError, 'overstated deleted-environment closure'):
                    self.check()
                self.proof = original

    def test_changed_remote_claim_or_results_ref_cannot_authorize_import(self):
        for name in ('remote_claim_ref', 'remote_results_ref'):
            with self.subTest(name=name):
                path = Path(self.proof[name]['path'])
                original = cloud.read(path)
                changed = deepcopy(original); changed['object']['sha'] = '0'*40
                write(path, changed); self.proof[name] = cloud.binding(path)
                with self.assertRaisesRegex(ValueError, 'ownership or closure ref changed'):
                    self.check()
                write(path, original); self.proof[name] = cloud.binding(path)

    def test_unrelated_deleted_thread_or_claim_worker_is_rejected(self):
        for name, key, value, error in [('app_unavailable', 'thread_id', 'unrelated', 'deletion has not been observed'),
                                       ('remote_claim', 'pid', 95786, 'does not match')]:
            with self.subTest(name=name):
                path = Path(self.proof[name]['path']); original = cloud.read(path)
                write(path, {**original, key: value}); self.proof[name] = cloud.binding(path)
                with self.assertRaisesRegex(ValueError, error):
                    self.check()
                write(path, original); self.proof[name] = cloud.binding(path)

    def test_unreviewed_supervisor_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'control flow changed'):
            cloud.check_cloud_closure(self.proof, self.folder)


class OriginalSnapshotSeamTests(unittest.TestCase):
    def test_historical_snapshot_audit_never_rewrites_current_moving_files(self):
        from tests.test_original_continuation import ContinuationTests as LocalFixture
        fixture = LocalFixture(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        names = ('latest-run.json', 'latest-checkpoint.json', 'sequence-ledger.json')
        original = {name: (fixture.root / name).read_text() for name in names}
        for name in names:
            write(fixture.root / name, {'current_import_boundary': True})
        before = {name: (fixture.root / name).read_bytes() for name in names}
        parent, remaining, proof = runner._local_original_continuation(
            fixture.anchor, fixture.config, moving_snapshots=original)
        self.assertEqual((parent, remaining, proof['next_sequence']), (fixture.parent, 4720, 337))
        self.assertEqual(before, {name: (fixture.root / name).read_bytes() for name in names})
        with self.assertRaisesRegex(ValueError, 'exact original moving snapshots'):
            runner._local_original_continuation(fixture.anchor, fixture.config,
                                                moving_snapshots={'latest-run.json': '{}'})


if __name__ == '__main__':
    unittest.main()
