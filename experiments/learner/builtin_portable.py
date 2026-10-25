"""Explicit, committed cross-platform restoration; never re-admit old histories.

The manifest pins both runtime inventories and exact glue-only source changes.
Historical JSON/checkpoint metadata is read verbatim, with paths resolved through
an explicit relocation map only when accessing an artifact. Fresh captures still
use the ordinary current-runtime admission and replay gates.
"""
from pathlib import Path
import subprocess

from experiments.headless.contextual_evidence import strict_json

SCHEMA = 'heroes-portable-resume-v1'
GLUE = frozenset(('builtin_training.py', 'builtin_training_resume.py',
                  'builtin_preservation.py', 'builtin_gameplay.py', 'checkpoints.py', 'builtin_first_run.py'))


class Migration:
    def __init__(self, path, root):
        from .builtin_training_resume import require, binding
        self.root = Path(root).resolve()
        self.path = Path(path).resolve()
        require(self.path.is_relative_to(self.root / 'configs'), 'Migration must be a committed config')
        relative = self.path.relative_to(self.root).as_posix()
        saved = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=self.root)
        require(saved == self.path.read_bytes(), 'Migration manifest is not exactly committed')
        self.value = strict_json(saved)
        require(self.value.get('schema') == SCHEMA, 'Invalid portable resume schema')
        self.binding = binding(self.path)
        require(self.value.get('historical_training_admission') is False,
                'Historical captures cannot be admitted by a platform migration')
        self.validate_source_changes()
        self.old_root = Path(self.value['historical_artifact_root'])
        self.new_root = Path(self.value['artifact_root']).resolve()
        require(self.old_root.is_absolute() and self.new_root.is_absolute(), 'Absolute artifact roots required')

    def validate_source_changes(self):
        from .builtin_training_resume import require
        for name in self.value['source_changes']:
            require(name.startswith('experiments/learner/') and Path(name).name in GLUE,
                    'Migration cannot change policy, features or PPO math')

    def resolve(self, path):
        from .builtin_training_resume import require
        path = Path(path)
        require(path.is_absolute() and '..' not in path.parts, 'Unsafe historical artifact path')
        require(path.is_relative_to(self.old_root), 'Historical artifact outside the declared root')
        target = (self.new_root / path.relative_to(self.old_root)).resolve()
        require(target.is_relative_to(self.new_root), 'Relocated artifact escapes its root')
        return target

    def artifact(self, bound):
        from .builtin_training_resume import binding, require
        require(set(bound) == {'path', 'sha256', 'bytes'}, 'Invalid historical binding')
        actual = binding(self.resolve(bound['path']))
        require(all(actual[k] == bound[k] for k in ('sha256', 'bytes')), 'Relocated artifact bytes changed')
        return actual

    @property
    def checkpoint_options(self):
        return {'historical_torch_version': self.value['from_source']['torch']}

    def source(self, parent, current):
        from .builtin_training_resume import require
        for key in ('python', 'torch', 'interface'):
            require(parent[key] == self.value['from_source'][key]
                    and current[key] == self.value['to_source'][key], 'Unpinned platform/schema transition: ' + key)
        require(parent['interface'] == current['interface'], 'Learned interface must remain unchanged')

    def runtime(self, parent_runtime, parent_native, current_runtime, current_native):
        from .builtin_training_resume import require
        expected = {'from_runtime': parent_runtime, 'from_native': parent_native,
                    'to_runtime': current_runtime, 'to_native': current_native}
        require(self.value['runtime_transition'] == expected, 'Unpinned native platform transition')
        require(parent_native['map'] == current_native['map'], 'Migration must retain the exact scenario')
        return {'unchanged': False, 'portable_migration': self.binding, **expected}

    def parent(self, plan_binding, report_binding, saved, progress, steps):
        from .builtin_training_resume import require
        actual = {'plan_sha256': plan_binding['sha256'], 'report_sha256': report_binding['sha256'],
                  'resume_sha256': saved['resume']['sha256'], 'policy_sha256': saved['policy']['sha256'],
                  'progress': vars(progress), 'optimizer_steps': steps}
        require(actual == self.value['parent'], 'Migration parent state differs from its exact pinned identity')

    def training_gate(self, runtime):
        from .builtin_training_resume import require, binding
        gate = self.value.get('linux_gate')
        require(type(gate) is dict and set(gate) == {'plan', 'report'},
                'Fresh Linux evaluation and replay gates must precede migration training')
        for value in gate.values():
            require(binding(value['path']) == value, 'Linux gate bytes changed')
        plan = strict_json(Path(gate['plan']['path']).read_bytes())
        report = strict_json(Path(gate['report']['path']).read_bytes())
        require(plan['schema'] == 'heroes-builtin-checkpoint-evaluation-v1-plan'
                and plan['runtime'] == runtime and plan['mode'] == 'evaluation'
                and plan['native'] == self.value['runtime_transition']['to_native']
                and plan['parent']['plan']['sha256'] == self.value['parent']['plan_sha256']
                and plan['checkpoint']['sha256'] == self.value['parent']['policy_sha256']
                and report['schema'] == 'heroes-builtin-checkpoint-evaluation-v1-result'
                and report['plan_sha256'] == gate['plan']['sha256']
                and report['status'] == 'finished' and report['worker_complete'] is True
                and report['supervisor_cleanup_verified'] is True
                and report.get('within_shared_budget') is True
                and report['optimizer_updates'] == report['optimizer_steps'] == 0
                and report['training_performed'] is False
                and len(report['attempts']) == len(plan['schedule']) == 2,
                'Linux gate must be a clean current-runtime evaluation on both seats')
        require([row['learner_player'] for row in plan['schedule']] == [0, 1], 'Linux gate seats differ')
        for expected, row in zip(plan['schedule'], report['attempts']):
            require(all(row.get(k) == v for k,v in expected.items())
                    and row['status'] == 'passed' and row['training_samples'] == 0
                    and all(row.get(k) is True for k in ('completion_receipt', 'cleanup_verified',
                                                        'identity_verified', 'replay_verified')),
                    'Linux gate contains an unverified capture/replay')
        return gate
