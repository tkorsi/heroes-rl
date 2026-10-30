"""Disposable freeze-receipt controls; fake files never represent native validation."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.episodes import scoring
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner.contextual import OWN_ARMY_INFORMATION
from tools import freeze_local_runtime as freezer, run_tiny_selfplay_one_game as tiny


class PassiveArmyRuntimeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='heroes-passive-runtime-fixture-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.runtime = self.root / 'runtime'; binary = self.runtime / 'bin'
        artifacts = []
        for name in freezer.ARTIFACTS:
            path = binary / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('not executable; synthetic runtime fixture ' + name).encode())
            value = tiny.binding(path)
            artifacts.append(dict(path=name, frozen_sha256=value['sha256'], bytes=value['bytes'],
                                  ad_hoc_signature_verified=True))
        (binary / 'Mods').mkdir()
        (binary / 'config').mkdir()
        resource = binary / 'config/settings.json'; resource.write_text('{}')
        self.original_map = self.root / 'original.vmap'; self.original_map.write_bytes(b'synthetic-map-only')
        game_map = self.runtime / 'maps/original.vmap'; game_map.parent.mkdir(); game_map.write_bytes(self.original_map.read_bytes())
        self.external = self.root / 'synthetic-external.dylib'; self.external.write_bytes(b'external-fixture')
        dependency = self.runtime / 'dependency-check.stdout'; dependency.write_text('synthetic fixture')
        self.recipe = {'engine/fixture.cpp': 'a' * 64}
        self.implementation = {'engine/fixture.cpp': 'a' * 64, 'python/fixture.py': 'b' * 64}
        self.freeze = dict(schema='heroes-local-runtime-freeze-v2', source_recipe_check=True, source_dirty=False,
            donor_load_references_removed=True, old_runtime_modified=False, gameplay_started=False, training_started=False,
            frozen_bin=str(binary), source_inputs={key: {'sha256': value} for key, value in self.recipe.items()},
            artifacts=artifacts, resources=[dict(path='config/settings.json', **freezer.identity(resource))],
            external_dependencies=[dict(resolved=str(self.external), **tiny.binding(self.external))],
            dependency_check=freezer.identity(dependency), map=tiny.binding(game_map))
        self.runtime_identity = runtime_identity(binary)
        self.native = dict(binary=str(binary / 'vcmiclient'), binary_sha256=tiny.binding(binary / 'vcmiclient')['sha256'],
            explorer_rl_sha256=self.runtime_identity['AI/libExplorerRL.dylib'], map={'sha256': self.freeze['map']['sha256']})
        self.preflight = dict(schema='heroes-local-runtime-preflight-v2', launcher_preflight_passed=True,
            client_help_exit_code=0, donor_load_detected=False, gameplay_started=False, training_started=False,
            runtime=self.runtime_identity, native=self.native, map=self.freeze['map'],
            actual_client_libvcmi_load=[str(binary / 'libvcmi.dylib')])
        self.validation = dict(schema='heroes-passive-armies-native-v1', status='passed',
            army_information=OWN_ARMY_INFORMATION, policy_support=tiny.GUARD, runtime=self.runtime_identity,
            source_implementation=self.implementation, players=[0, 1], controls=2, native_replay_verified=True,
            both_policies_reconstructed=True, owned_cleanup_verified=True, optimizer_updates=0, training_samples_admitted=0)
        self.validation_path = self.root / 'synthetic-validation.json'
        self.write_receipts()
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(tiny, 'RUNTIME', self.runtime))
        self.stack.enter_context(patch.object(tiny, 'RUNTIME_FREEZE_SHA', tiny.binding(self.runtime / 'freeze-receipt.json')['sha256']))
        self.stack.enter_context(patch.object(tiny, 'RUNTIME_PREFLIGHT_SHA', tiny.binding(self.runtime / 'preflight.json')['sha256']))
        self.stack.enter_context(patch.object(tiny, 'REQUIRED_NATIVE_VALIDATION', self.validation_path))
        self.stack.enter_context(patch.object(tiny, 'REQUIRED_NATIVE_VALIDATION_SHA', tiny.binding(self.validation_path)['sha256']))
        self.stack.enter_context(patch.object(scoring, 'engine_recipe_digests', return_value=self.recipe))
        self.stack.enter_context(patch.object(tiny.evaluation, 'source_identity', return_value={'implementation': self.implementation}))

    def write_receipts(self):
        tiny.write(self.runtime / 'freeze-receipt.json', self.freeze)
        tiny.write(self.runtime / 'preflight.json', self.preflight)
        tiny.write(self.validation_path, self.validation)

    def inspect(self):
        return tiny.inspect_runtime({'game_map': str(self.original_map)})

    def test_new_runtime_selection_protects_every_frozen_input_without_mutating_baseline(self):
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in self.root.rglob('*') if path.is_file()}
        result = self.inspect()
        self.assertEqual(result['binary'], str(self.runtime / 'bin/vcmiclient'))
        self.assertEqual(result['army_information'], OWN_ARMY_INFORMATION)
        self.assertEqual(result['runtime'], self.runtime_identity)
        self.assertEqual(result['native'], self.native)
        protected = result['runtime_protected']
        self.assertIn(str(self.external), protected)
        self.assertIn(str(self.runtime / 'bin/config/settings.json'), protected)
        self.assertIn(str(self.validation_path), protected)
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in self.root.rglob('*') if path.is_file()})

    def test_changed_binary_resource_external_library_or_map_rejects(self):
        for path in (self.runtime / 'bin/AI/libExplorerRL.dylib', self.runtime / 'bin/config/settings.json',
                     self.external, self.original_map):
            before = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(before + b'changed')
                try:
                    with self.assertRaises(ValueError): self.inspect()
                finally: path.write_bytes(before)

    def test_unreviewed_source_or_incomplete_native_acceptance_rejects(self):
        for changes in ({'status': 'failed'}, {'players': [0]}, {'native_replay_verified': False},
                        {'both_policies_reconstructed': False}, {'owned_cleanup_verified': False},
                        {'training_samples_admitted': 1}, {'optimizer_updates': 1}, {'controls': 0},
                        {'source_implementation': {}}, {'runtime': {}}, {'army_information': None}):
            with self.subTest(changes=changes):
                tiny.write(self.validation_path, {**self.validation, **changes})
                with patch.object(tiny, 'REQUIRED_NATIVE_VALIDATION_SHA', tiny.binding(self.validation_path)['sha256']), \
                     self.assertRaisesRegex(ValueError, 'two-seat native validation'):
                    self.inspect()
        tiny.write(self.validation_path, self.validation)
        with patch.object(scoring, 'engine_recipe_digests', return_value={'engine/fixture.cpp': 'c' * 64}), \
             self.assertRaisesRegex(ValueError, 'engine recipe'):
            self.inspect()

    def test_extra_resource_and_dirty_or_wrong_loader_receipts_reject(self):
        added = self.runtime / 'bin/config/unrecorded.json'; added.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'resource inventory'): self.inspect()
        added.unlink()
        changed = deepcopy(self.freeze); changed['source_dirty'] = True
        tiny.write(self.runtime / 'freeze-receipt.json', changed)
        with patch.object(tiny, 'RUNTIME_FREEZE_SHA', tiny.binding(self.runtime / 'freeze-receipt.json')['sha256']), \
             self.assertRaisesRegex(ValueError, 'freeze is incomplete'):
            self.inspect()
        tiny.write(self.runtime / 'freeze-receipt.json', self.freeze)
        changed = deepcopy(self.preflight); changed['actual_client_libvcmi_load'] = ['/unrelated/libvcmi.dylib']
        tiny.write(self.runtime / 'preflight.json', changed)
        with patch.object(tiny, 'RUNTIME_PREFLIGHT_SHA', tiny.binding(self.runtime / 'preflight.json')['sha256']), \
             self.assertRaisesRegex(ValueError, 'loader preflight'):
            self.inspect()


if __name__ == '__main__':
    unittest.main()
