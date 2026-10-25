"""A verifier repair cannot silently admit policy or native gameplay changes."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiments.learner import builtin_training_resume as resume

class VerifierTransitionTests(unittest.TestCase):
    def setUp(self):
        self.old_runtime={'vcmireplay':'old','libvcmi.dylib':'rules'}
        self.new_runtime={'vcmireplay':'new','libvcmi.dylib':'rules'}
        self.old_native={'binary':'/durable/v2/vcmiclient','binary_sha256':'client','map':{'sha256':'map'}}
        self.new_native={**self.old_native,'binary':'/durable/v3/vcmiclient'}
        self.transition={'from_runtime':self.old_runtime,'to_runtime':self.new_runtime,
                         'from_native':self.old_native,'to_native':self.new_native}

    def test_only_the_exact_declared_verifier_transition_passes(self):
        with patch.object(resume,'transition_manifest',return_value={'runtime_changes':[self.transition]}):
            self.assertFalse(resume.compatible_runtime(self.old_runtime,self.old_native,self.new_runtime,self.new_native)['unchanged'])
            changed={**self.new_runtime,'vcmireplay':'unknown'}
            with self.assertRaisesRegex(ValueError,'Unreviewed'):
                resume.compatible_runtime(self.old_runtime,self.old_native,changed,self.new_native)

    def test_even_a_declared_gameplay_change_is_rejected(self):
        for which in ('rules','client'):
            transition=deepcopy(self.transition)
            if which=='rules':transition['to_runtime']['libvcmi.dylib']='different'
            else:transition['to_native']['binary_sha256']='different'
            with patch.object(resume,'transition_manifest',return_value={'runtime_changes':[transition]}):
                with self.assertRaisesRegex(ValueError,'Only an exact reviewed verifier'):
                    resume.compatible_runtime(transition['from_runtime'],transition['from_native'],
                                              transition['to_runtime'],transition['to_native'])

    def test_committed_manifest_cannot_allow_model_source_and_mutation_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'configs/builtin-resume-transitions-v1.json';p.parent.mkdir()
            value={'schema':'heroes-reviewed-resume-transitions-v1','source_changes':{'experiments/learner/contextual_model.py':{'from':'old','to':'new'}},'runtime_changes':[]}
            p.write_text(json.dumps(value))
            for args in (('init','-q'),('add','.'),('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')):
                subprocess.run(['git',*args],cwd=root,check=True,capture_output=True)
            with patch.object(resume,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'cannot change policy'):resume.transition_manifest()
                p.write_text('{}')
                with self.assertRaisesRegex(ValueError,'not exactly committed'):resume.transition_manifest()
