"""Lossless external-history storage and rejection controls; no native games."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.learner import contextual as c, contextual_storage as storage
from experiments.learner.contextual_model import ContextualActorCritic, ModelConfig, model_schema_for
from experiments.learner.contextual_rollout import ContextualCollector
from experiments.learner.interaction_guard import REVEALED_VERSION
from tests.test_contextual_rollout import initial, request, resolve, finish_pair
from tests.test_terminal_rollout import collect as terminal_collect


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


class ContextualStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='heroes-storage-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        torch.manual_seed(91)
        self.model = ContextualActorCritic(ModelConfig(hidden_size=8, feature_size=4))

    def fixture(self, boundary='natural', *, trajectory=c.TERMINAL_TRAJECTORY_SCHEMA,
                terminal_closure=False, purpose='synthetic', schema=c.TOWN_SCHEMA, guard_version=None):
        folder = Path(tempfile.mkdtemp(dir=self.root))
        if terminal_closure:
            collected, _, _ = terminal_collect(self.model, trajectory_schema=trajectory, purpose=purpose)
        else:
            model = self.model if schema == c.TOWN_SCHEMA else ContextualActorCritic(
                ModelConfig(schema=model_schema_for(schema), hidden_size=8, feature_size=4))
            options = {} if guard_version is None else {'guard_version': guard_version}
            collector = ContextualCollector((model, model), ('red-initial', 'blue-initial'),
                learner_player=0, opponent_pool='synthetic-storage-pool', purpose=purpose, origin='synthetic',
                interface=c.profile_for(schema).environment, trajectory_schema=trajectory, **options)
            collector.start('synthetic-collector')
            histories = tuple(initial(player, schema=schema) for player in (0, 1))
            for history in histories:
                resolve(history, collector.act(request(history)))
            end = finish_pair(histories, boundary)
            if c.profile_for(schema).dialogs:
                end.manifest['dialog_cutoff_drain'] = None
            collected = collector.finish(end, histories=tuple(history.frames for history in histories))
        manifest = json.loads(collected.manifest_json)
        for color, rollout in zip(('red', 'blue'), collected.rollouts):
            profile = c.profile_for(rollout.history[0].schema)
            path = folder / 'agents' / color / profile.history_filename
            path.parent.mkdir(parents=True)
            payload = b''.join((c.canonical_frame(frame) + '\n').encode() for frame in rollout.history)
            path.write_bytes(payload)
            manifest['players'][color][profile.history_key].update(
                path=str(path.relative_to(folder)), sha256=hashlib.sha256(payload).hexdigest())
        (folder / 'manifest.json').write_bytes(encode(manifest))
        collected = replace(collected, manifest_json=json.dumps(manifest, sort_keys=True, separators=(',', ':')))
        path = folder / 'neural-rollout.json'
        return path, collected

    def saved(self, **kwargs):
        path, collected = self.fixture(**kwargs)
        storage.save_collected(path, collected)
        return path, collected, json.loads(path.read_bytes())

    def test_external_and_legacy_match_every_boundary_and_trajectory_version(self):
        for trajectory in (c.TRAJECTORY_SCHEMA, c.TERMINAL_TRAJECTORY_SCHEMA):
            for boundary in ('natural', 'cutoff', 'failure'):
                with self.subTest(trajectory=trajectory, boundary=boundary):
                    path, collected, raw = self.saved(boundary=boundary, trajectory=trajectory)
                    self.assertEqual(storage.load_collected(path), collected)
                    legacy = path.with_name('legacy.json'); legacy.write_bytes(encode(collected.to_dict()))
                    self.assertEqual(storage.load_collected(legacy), collected)
                    self.assertEqual(raw['schema'], storage.SCHEMA)
                    self.assertTrue(all('history' not in entry['records'] for entry in raw['rollouts']))
                    if boundary == 'failure':
                        self.assertFalse(any(rollout.optimization_eligible for rollout in
                                             storage.load_collected(path).rollouts))

    def test_expanded_revealed_guard_and_exact_support_survive(self):
        path, collected, raw = self.saved(schema=c.EXPANDED_SCHEMA, guard_version=REVEALED_VERSION)
        restored = storage.load_collected(path)
        self.assertEqual(restored, collected)
        self.assertTrue(all(rollout.decisions[0].support.guard.version == REVEALED_VERSION
                            for rollout in restored.rollouts))
        raw['rollouts'][0]['records']['decisions'][0]['support']['interaction_guard']['version'] = 'invented'
        path.write_bytes(encode(raw))
        with self.assertRaises(ValueError): storage.load_collected(path)

    def test_terminal_closure_and_diagnostic_exclusion_survive_exactly(self):
        path, collected, _ = self.saved(terminal_closure=True, purpose='diagnostic')
        restored = storage.load_collected(path)
        self.assertEqual(restored, collected)
        for rollout in restored.rollouts:
            self.assertIsNotNone(rollout.samples[-1].terminal_closure)
            self.assertFalse(rollout.optimization_eligible)

    def test_envelope_records_and_manifest_tampering_fail_closed(self):
        changes = {
            'schema': lambda raw: raw.update(schema='unsupported'),
            'artifact_schema': lambda raw: raw.update(artifact_schema='unsupported'),
            'extra_field': lambda raw: raw.update(unexpected=True),
            'missing_player': lambda raw: raw['rollouts'].pop(),
            'reversed_players': lambda raw: raw['rollouts'].reverse(),
            'manifest': lambda raw: raw['manifest'].update(episode='another-episode'),
            'history_ref_digest': lambda raw: raw['rollouts'][0]['history_ref'].update(sha256='a' * 64),
            'history_ref_size': lambda raw: raw['rollouts'][0]['history_ref'].update(bytes=1),
            'history_ref_canonical': lambda raw: raw['rollouts'][0]['history_ref'].update(canonical_sha256='a' * 64),
            'unknown_record': lambda raw: raw['rollouts'][0]['records'].update(unexpected=True),
            'history_digest': lambda raw: raw['rollouts'][0]['records'].update(history_sha256='a' * 64),
            'role': lambda raw: raw['rollouts'][0]['records'].update(role='frozen_opponent'),
            'sample_reward': lambda raw: raw['rollouts'][0]['records']['samples'][0].update(reward=-1),
            'sample_sequence': lambda raw: raw['rollouts'][0]['records']['samples'][0].update(result_seq=1),
            'estimate': lambda raw: raw['rollouts'][0]['records']['estimates'][0].update(value=123),
            'support': lambda raw: raw['rollouts'][0]['records']['decisions'][0]['support'].update(context='altered'),
            'prefix': lambda raw: raw['rollouts'][0]['records']['decisions'][0]['prefix'].update(history_sha256='a' * 64),
            'conditional': lambda raw: raw['rollouts'][0]['records']['decisions'][0]['conditionals'][0].update(selected_index=999),
        }
        path, _, original = self.saved()
        for name, mutate in changes.items():
            with self.subTest(mutation=name):
                raw = deepcopy(original); mutate(raw); path.write_bytes(encode(raw))
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    storage.load_collected(path)

    def rebind_file(self, path, raw, payload):
        """Refresh only byte bindings to expose strict semantic validation."""
        reference = raw['rollouts'][0]['history_ref']
        history = path.parent / reference['path']; history.write_bytes(payload)
        reference.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        manifest = raw['manifest']
        key = c.profile_for(manifest['environment']).history_key
        manifest['players']['red'][key]['sha256'] = reference['sha256']
        (path.parent / 'manifest.json').write_bytes(encode(manifest)); path.write_bytes(encode(raw))

    def test_history_truncation_extra_reorder_alteration_and_nonfinite_reject_even_with_rebound_bytes(self):
        mutations = {
            'truncated_last_line': lambda lines: b''.join(lines).rstrip(b'\n'),
            'missing_frame': lambda lines: b''.join(lines[:-1]),
            'extra_frame': lambda lines: b''.join(lines + [lines[-1]]),
            'reordered': lambda lines: b''.join([lines[1], lines[0], *lines[2:]]),
            'duplicate_key': lambda lines: b'{"seq":0,' + lines[0][1:] + b''.join(lines[1:]),
            'nonfinite': lambda lines: lines[0].replace(b'"day":1', b'"day":NaN') + b''.join(lines[1:]),
            'changed_resource': lambda lines: lines[0].replace(b'"resources":[0,', b'"resources":[999,') + b''.join(lines[1:]),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                path, _, raw = self.saved()
                history = path.parent / raw['rollouts'][0]['history_ref']['path']
                payload = mutate(history.read_bytes().splitlines(keepends=True))
                self.assertNotEqual(payload, history.read_bytes(), 'Mutation must change the fixture')
                self.rebind_file(path, raw, payload)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    storage.load_collected(path)

    def test_unbound_history_or_manifest_change_rejects(self):
        for target in ('history', 'manifest'):
            with self.subTest(target=target):
                path, _, raw = self.saved()
                changed = (path.parent / raw['rollouts'][0]['history_ref']['path']
                           if target == 'history' else path.parent / 'manifest.json')
                changed.write_bytes(changed.read_bytes() + b' ')
                if target == 'manifest':
                    manifest = json.loads(changed.read_bytes()); manifest['episode'] = 'another'
                    changed.write_bytes(encode(manifest))
                with self.assertRaises(ValueError): storage.load_collected(path)

    def test_history_paths_cannot_escape_or_follow_symlinks(self):
        for name in ('../outside.jsonl', '/outside.jsonl', 'agents/red/link.jsonl'):
            with self.subTest(path=name):
                path, _, raw = self.saved()
                reference = raw['rollouts'][0]['history_ref']
                original = path.parent / reference['path']
                if name.endswith('link.jsonl'):
                    (path.parent / name).symlink_to(original)
                reference['path'] = name
                manifest = raw['manifest']; key = c.profile_for(manifest['environment']).history_key
                manifest['players']['red'][key]['path'] = name
                (path.parent / 'manifest.json').write_bytes(encode(manifest)); path.write_bytes(encode(raw))
                with self.assertRaisesRegex(ValueError, 'path|symlink'):
                    storage.load_collected(path)

    def test_strict_json_rejects_duplicate_and_nonfinite_envelope_fields(self):
        path, _, _ = self.saved(); original = path.read_bytes()
        for payload in (b'{"schema":"duplicate",' + original[1:],
                        original.replace(b'"audit":{', b'"audit":{"invalid":NaN,')):
            with self.subTest(payload=payload[:30]):
                self.assertNotEqual(payload, original)
                path.write_bytes(payload)
                with self.assertRaises(ValueError): storage.load_collected(path)

    def test_existing_target_or_racing_writer_is_never_overwritten(self):
        path, collected, _ = self.saved(); original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'replace'):
            storage.save_collected(path, collected)
        self.assertEqual(path.read_bytes(), original)
        path.unlink()
        real_link = storage.os.link

        def race(source, target):
            Path(target).write_bytes(b'other writer')
            real_link(source, target)

        with patch.object(storage.os, 'link', side_effect=race), self.assertRaises(FileExistsError):
            storage.save_collected(path, collected)
        self.assertEqual(path.read_bytes(), b'other writer')
        self.assertEqual(list(path.parent.glob('.rollout-*.partial')), [])

    def test_failed_write_or_sync_leaves_no_artifact_or_temporary_file(self):
        for operation in ('dump', 'fsync'):
            with self.subTest(operation=operation):
                path, collected = self.fixture()
                target = storage.json if operation == 'dump' else storage.os
                with patch.object(target, operation, side_effect=OSError('injected storage failure')), \
                     self.assertRaisesRegex(OSError, 'injected'):
                    storage.save_collected(path, collected)
                self.assertFalse(path.exists())
                self.assertEqual(list(path.parent.glob('.rollout-*.partial')), [])

    def test_atomic_publication_exposes_complete_synced_payload_only(self):
        path, collected = self.fixture(); real_link = storage.os.link

        def observe(source, target):
            self.assertFalse(Path(target).exists())
            raw = json.loads(Path(source).read_bytes())
            self.assertEqual(raw['schema'], storage.SCHEMA)
            self.assertEqual(raw['manifest'], json.loads(collected.manifest_json))
            real_link(source, target)

        with patch.object(storage.os, 'link', side_effect=observe) as publish:
            storage.save_collected(path, collected)
        publish.assert_called_once()
        self.assertEqual(storage.load_collected(path), collected)
        self.assertEqual(list(path.parent.glob('.rollout-*.partial')), [])


if __name__ == '__main__':
    unittest.main()
