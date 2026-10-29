"""Read the pinned Mac handoff through an explicit, byte-preserving resolver.

Historical bindings stay logical Mac paths. Only filesystem access is relocated;
new Linux descriptors explicitly bind the actual copies. No JSON is rewritten.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath

MANIFEST_SHA256 = '4849dec40e2e82a0911196e33c845dab6f7ea1db769e23b89118b9808f67e27f'
FENCE_SHA256 = 'fa47da0413e261c0118aaf5711519897cbb01caceb987a74c3c35031c0e30322'
HANDOFF = 'checkpoint1229-cloud-a08f55ff0c7548f58ff9d50ef5af8ba7'
FULL_STATE = 'c5882092150d55b4c07df368534f63b0c742bdd74982f1228210007f67c5ad7e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Evidence:
    def __init__(self, bundle, extracted):
        from experiments.headless.contextual_evidence import strict_json
        self.bundle, self.extracted = Path(bundle).resolve(), Path(extracted).resolve()
        require(digest(self.bundle/'manifest.json') == MANIFEST_SHA256, 'Unreviewed lineage manifest')
        self.manifest = strict_json((self.bundle/'manifest.json').read_bytes())
        require(self.manifest['schema'] == 'heroes-private-full-continuation-evidence-v1'
                and self.manifest['checkpoint'] == 1229 and self.manifest['full_state_sha256'] == FULL_STATE,
                'Wrong complete-state handoff')
        receipt = strict_json((self.extracted/'extraction-receipt.json').read_bytes())
        require(receipt['verified'] is True and receipt['manifest_sha256'] == MANIFEST_SHA256
                and receipt['source_paths'] == 5241 and receipt['unique_objects'] == 4482
                and receipt['package_count'] == 37 and receipt['output'] == str(self.extracted),
                'Incomplete or relocated extraction receipt')
        self.files = {}
        for row in self.manifest['files']:
            name = row['original_path']
            p = PurePosixPath(name)
            require(p.is_absolute() and str(p) == name and '..' not in p.parts
                    and row['restored_path'] == 'tree/'+name[1:] and name not in self.files,
                    'Unsafe or duplicate historical path')
            self.files[name] = row
        require(len(self.files) == 5241, 'Incomplete path inventory')
        self.fence()

    def resolve(self, path):
        name = str(path)
        require(name in self.files, 'Historical file absent from pinned handoff: '+name)
        actual = self.extracted/self.files[name]['restored_path']
        require(actual.resolve() == actual and actual.is_file() and not actual.is_symlink(),
                'Relocated evidence is missing or redirected: '+name)
        return actual

    def binding(self, path):
        name = str(path)
        if name not in self.files:
            require(not name.startswith('/Users/'), 'Unbound historical artifact: '+name)
            actual = Path(path).resolve()
        else:
            actual = self.resolve(name)
        result = dict(path=name if name in self.files else str(actual),
                      bytes=actual.stat().st_size, sha256=digest(actual))
        if name in self.files:
            expected = self.files[name]
            require(all(result[k] == expected[k] for k in ('bytes', 'sha256')),
                    'Historical artifact bytes changed: '+name)
        return result

    def read_bytes(self, path):
        self.binding(path)
        return (self.resolve(path) if str(path) in self.files else Path(path)).read_bytes()

    def read(self, path):
        from experiments.headless.contextual_evidence import strict_json
        return strict_json(self.read_bytes(path))

    def rebind(self, descriptor):
        require(set(descriptor) == {'plan', 'report', 'policy', 'resume', 'initial_checkpoint'},
                'Complete five-file descriptor required')
        result = {}
        for key, value in descriptor.items():
            require(self.binding(value['path']) == value, 'Historical descriptor binding changed')
            result[key] = {**value, 'path': str(self.resolve(value['path']))}
        return result

    def descriptor(self, descriptor):
        from experiments.learner.local_script_training import read_descriptor
        return read_descriptor(self.rebind(descriptor))

    def inventory(self, root, pattern='*'):
        root = Path(root)
        actual = self.extracted/'tree'/str(root).lstrip('/')
        require(actual.resolve() == actual and actual.is_dir(), 'Missing historical inventory root')
        result = []
        for path in sorted(actual.rglob(pattern)):
            require(not path.is_symlink(), 'Historical inventory symlink')
            if path.is_file():
                logical = root/path.relative_to(actual)
                result.append(self.binding(logical))
        return result

    def fence(self):
        from experiments.headless.contextual_evidence import strict_json
        path = self.bundle/'handoff-control.json'
        require(path.stat().st_size == 4273 and digest(path) == FENCE_SHA256,
                'Mac fence receipt changed')
        value = strict_json(path.read_bytes())
        require(value['schema'] == 'heroes-mac-cloud-handoff-fence-v1'
                and value['handoff_id'] == HANDOFF and value['checkpoint'] == 1229
                and value['remaining_games'] == 2600 and value['next_sequence'] == 869
                and value['status'] == 'mac_launches_fenced_pending_cloud_handoff'
                and value['current_mode'] == '0444' and value['locks_held_through_fence'] is True
                and value['write_open_denied'] is True and value['official_file_contents_changed'] is False
                and value['signals_sent'] == 0 and value['training_launched'] is False
                and len(value['locks']) == 2 and all(row['uid'] == 501 and row['original_mode'] == '0644'
                                                  for row in value['locks'])
                and all(row['a'] == 'r' for row in value['before_open_descriptors']
                        + value['after_open_descriptors']), 'Incomplete Mac launch exclusion proof')
        return value

    def process_closure(self, deadlines, run):
        # Foreign Mac PID numbers do not identify Linux processes. Revalidate
        # every historical deadline and the concrete cooperative-user fence.
        fence = self.fence()
        for item in deadlines:
            require(self.binding(item['path']) == item, 'Historical deadline changed')
            rows = [json.loads(line) for line in self.read_bytes(item['path']).splitlines()]
            require(rows and all(r.get('schema') == 'heroes-process-deadline-v1' for r in rows),
                    'Malformed historical deadline ledger')
        require(fence['seed_ledger'] == self.binding(fence['seed_ledger']['path']), 'Fenced ledger changed')
        return dict(method='durable_mac_write_open_fence', handoff_id=HANDOFF,
                    fence_sha256=FENCE_SHA256, foreign_pid_numbers_signalled=False, signals_sent=0)

    @contextmanager
    def recovery_access(self):
        from tools import recover_interrupted_training as recovery
        token = recovery.ARTIFACT_ACCESS.set(self)
        try:
            yield
        finally:
            recovery.ARTIFACT_ACCESS.reset(token)
