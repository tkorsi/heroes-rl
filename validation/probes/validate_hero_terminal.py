#!/usr/bin/env python3
"""Both-color natural-boundary diagnostics on the exact Tavern-enabled scenario.

Uses the existing explicit town-capture/seven-day mechanics controls. These are
scripted diagnostic actions, never learner data or ordinary-policy statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from experiments.explorer.create_hero_scenario import verify_hero_scenario
from experiments.headless.contextual_evidence import runtime_identity
from experiments.learner import contextual as c
from experiments.learner.first_run import implementation_identity
from validation.probes import validate_contextual_native as base
from validation.probes.validate_town_capture import CaptureRun, capture_case

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=102201)
    parser.add_argument("--interface", choices=(c.HEROES_ENVIRONMENT, c.MANAGEMENT_ENVIRONMENT, c.DIALOGS_ENVIRONMENT), default=c.HEROES_ENVIRONMENT)
    args = parser.parse_args(argv)
    if not 1 <= args.seed < 2147483647:
        parser.error('Use a positive bounded native seed')
    for name in ('binary', 'data', 'output'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output.exists(): parser.error('Preserve every attempt; output exists')
    args.map = ROOT / 'experiments/explorer/maps/hero-management-v1.vmap'
    verify_hero_scenario(args.map)
    profile = c.profile_for(args.interface)
    args.max_days, args.max_decisions = 20, 100
    args.case_seconds, args.watchdog, args.startup_seconds = 60, 20, 30
    env = base.load('heroes_terminal_probe_env', ROOT / 'experiments/headless/env.py')
    auditor = base.load('heroes_terminal_probe_auditor', ROOT / 'validation/probes/audit_contextual_histories.py')
    env.VcmiLauncher(args.binary, args.data, args.map).preflight()
    subprocess.run(['python3', str(ROOT / 'tools/prepare_vcmi.py'), '--check'], cwd=ROOT, check=True)
    args.output.mkdir(parents=True)
    implementation = implementation_identity()
    base.write(args.output / 'plan.json', {
        'schema': f'heroes-v{profile.version}-natural-boundary-diagnostics-v1', 'interface': args.interface,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'implementation': implementation, 'native_runtime': runtime_identity(args.binary.parent),
        'map_sha256': base.sha(args.map), 'attacker_colors': list(base.COLORS),
        'seeds': [args.seed, args.seed + 1], 'bounds': {
            name: getattr(args, name) for name in ('max_days', 'max_decisions', 'case_seconds', 'watchdog')},
        'training_input': False, 'training_performed': False, 'ordinary_outcome_frequency_claim': False})
    reports = []
    for index, color in enumerate(base.COLORS):
        run = CaptureRun('capture-' + color, args, env, c, auditor, args.seed + index)
        error = None
        try:
            run.start()
            capture_case(run, color)
            run.persist_and_audit()
            base.ensure(implementation_identity() == implementation, 'Authored sources changed during validation')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            run.check('natural_boundary_execution', 'failed', error=error)
            try:
                if run.env.process is not None and run.end is None:
                    run.env._note_problem('protocol_error', error)
                    run.end = run.state = run.env._finish(None)
                if run.end is not None and not run.audit_attempted: run.persist_and_audit()
            except Exception as cleanup:
                run.check('failure_evidence_finalization', 'failed', error=f'{type(cleanup).__name__}: {cleanup}')
        finally:
            run.env.close()
        report = run.summary(error)
        report.update(attacker=color, training_input=False, ordinary_outcome_frequency_claim=False)
        reports.append(report)
        base.write(run.directory / 'report.json', report)
    report = {'schema': f'heroes-v{profile.version}-natural-boundary-diagnostics-v1', 'cases': reports,
              'training_performed': False, 'ordinary_outcome_frequency_claim': False}
    base.write(args.output / 'report.json', report)
    print(json.dumps({'statuses': [row['probe_status'] for row in reports], 'report': str(args.output / 'report.json')}))
    return 0 if all(row['probe_status'] == 'passed' for row in reports) else 1


if __name__ == '__main__': raise SystemExit(main())
