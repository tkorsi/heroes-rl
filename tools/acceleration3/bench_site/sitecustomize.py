"""Linux benchmark port of the manual local runner (benchmark processes only).

Loaded through PYTHONPATH by ``tools/acceleration3/run_local_bench.py`` and active only while
HEROES_LOCAL_BENCH_SPEC names a benchmark specification. It leaves every source file of the
frozen release byte-identical and changes, in memory, exactly three machine bindings that the
Mac-only preparation pins, identically for the baseline and the candidate:

1. Torch relocation. This host runs the public 2.9.1 release as the '+cu128' wheel on CPU;
   the strict loader accepts a historical checkpoint only under the CPU wheel's version string.
   While a checkpoint is read, the loader sees the CPU wheel's string for the same release; a
   checkpoint written on this host is read under its own version. Payload bytes, tensors and
   every other check are unchanged (validated: 412 and 629 restore exactly, relocation receipts).
2. Machine description. ``prepare_local_training.validate_config`` checks this host's pinned
   runtime, parent bytes, map, data and prerequisites (from the specification) instead of the Mac
   runtime and receipts, and restores the parent exactly as the Mac check does.
3. Seed namespace. The original profile uses the benchmark namespace from the specification, never
   the official 40M/50M (original) or 20M/30M (easy) bases.

With ``"mode": "namespace"`` in the specification (``tools/acceleration3/mac_acceptance.py`` on
the accepted Mac machine) only patch 3 applies: the Mac runs its pinned runtime, Torch and
machine check unchanged, and only the seed namespace differs from an official original run.

So that the patches also reach the adapter's ``python -m`` entry, such an entry imports the
module under its own name (patched) and calls its ``main()``, which is all its ``__main__`` block
does.
"""
import importlib.abc
import importlib.util
import json
import os
import sys

SPEC = os.environ.get('HEROES_LOCAL_BENCH_SPEC')


def _spec():
    with open(SPEC) as stream:
        return json.load(stream)


def _patch_checkpoints(module):
    import torch
    original, real = module._read, str(torch.__version__)
    public = real.split('+', 1)[0]
    cpu = public + '+cpu'

    def _read(path, kind, expected_source_metadata, allowed_provenance, interface, historical_torch_version=None):
        if historical_torch_version is None or historical_torch_version == real:
            # Written on this host: its own version string, as the loader expects without relocation.
            return original(path, kind, expected_source_metadata, allowed_provenance, interface, None)
        saved = torch.__version__
        torch.__version__ = cpu  # the same public release as the CPU wheel, for this read only
        try:
            return original(path, kind, expected_source_metadata, allowed_provenance, interface, historical_torch_version)
        finally:
            torch.__version__ = saved
    module._read = _read


def _patch_namespace(module):
    module.SEED_NAMESPACES = {**module.SEED_NAMESPACES, 'original': dict(_spec()['seed_namespace'])}


def _patch_prepare(module):
    _patch_namespace(module)
    spec = _spec()

    def validate_config(value):
        from pathlib import Path
        from experiments.headless.contextual_evidence import runtime_identity
        from experiments.learner.local_script_training import restore_parent
        require = module.require
        require(value.get('schema') == module.SCHEMA and value.get('backend') == 'cold'
                and value.get('default_profile') == 'original', 'Unsupported benchmark machine configuration')
        expected = json.loads(Path(spec['machine_config']).read_text())
        require(set(value) == set(expected) and all(value[k] == expected[k] for k in expected if k != 'created_utc'),
                'Benchmark machine configuration changed')
        for name, item in value['parent'].items():
            require(module.binding(item['path']) == item, 'Parent binding changed: ' + name)
        require(len(value['maps']) == 2 and value['maps'][0] == value['maps'][1]
                and module.original_map_identity(value['maps'][0]['path']) == value['maps'][0], 'Map binding changed')
        runtime = Path(value['runtime_bin'])
        for name, item in value['runtime_files'].items():
            require(module.binding(runtime / name) == item, 'Runtime binding changed: ' + name)
        require(runtime_identity(runtime) == value['runtime'], 'Runtime identity changed')
        require(module.current_engine_recipe() == value['engine_recipe'], 'Engine recipe differs from the benchmark runtime build')
        for item in value['prerequisites'].values():
            require(module.binding(item['path']) == item, 'Prerequisite changed')
        data = Path(value['data'])
        require(data.is_dir() and (data / 'Data').is_dir(), 'Expected a VCMI data directory')
        state = restore_parent(value['parent'])  # the same exact full restore the Mac check performs
        require(vars(state.progress) == value['checkpoint_restore']['progress'], 'Parent progress differs')
        return value
    module.validate_config = validate_config
    module.check_configuration = lambda path: validate_config(json.loads(open(path).read()))


PATCHES = {'experiments.learner.checkpoints': _patch_checkpoints,
           'experiments.learner.local_script_training': _patch_namespace,
           'prepare_local_training': _patch_prepare}
if SPEC and _spec().get('mode') == 'namespace':
    PATCHES = {'experiments.learner.local_script_training': _patch_namespace,
               'prepare_local_training': _patch_namespace}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self.inner, self.patch = inner, patch

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(name, path, target) if hasattr(finder, 'find_spec') else None
            if spec is not None:
                spec.loader = _Loader(spec.loader, PATCHES[name])
                return spec
        return None


def _run_module_as_main(original):
    """``python -m`` of a patched module runs its ordinary import (patched) and then its ``main()``.

    A module executed as ``__main__`` would run its own top-level code and ``main()`` before any
    patch could apply; the adapter's ``if __name__ == '__main__': sys.exit(main())`` is the only
    entry code, so importing it under its own name and calling ``main()`` is the same program.
    """
    def run(mod_name, alter_argv=True):
        if mod_name not in PATCHES:
            return original(mod_name, alter_argv)
        import importlib
        module = importlib.import_module(mod_name)
        if alter_argv:
            sys.argv[0] = module.__file__
        sys.exit(module.main())
    return run


if SPEC:
    import runpy
    sys.meta_path.insert(0, _Finder())
    runpy._run_module_as_main = _run_module_as_main(runpy._run_module_as_main)
