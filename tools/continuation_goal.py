"""Audit a fixed manual game goal against verified original-map ancestry.

The controller calls this while holding its campaign locks. A fresh invocation
counts only completed descendants; interrupted windows never reset the goal.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOALS = {'checkpoint1392-remaining772-v1': (
    'configs/checkpoint1392-remaining772.json',
    '3ceddf0c4a71d128e6c678b7d1fa6354fe3266bcc7c7b709c9e84087792440c2'),
    'checkpoint1585-remaining1176-v1': (
    'configs/checkpoint1585-remaining1176.json',
    '097e70bdacac96b456406577c1b6ba4cb30f6d1c4b1468f208b05264e8a7d917')}

FAILED_ANCHOR_SCHEMA = 'heroes-manual-continuation-goal-v1'
COMPLETED_ANCHOR_SCHEMA = 'heroes-completed-run-continuation-goal-v1'


def load_goal(name):
    from experiments.headless.contextual_evidence import strict_json
    from experiments.learner.builtin_training_resume import binding, require
    require(name in GOALS, 'Unknown manual continuation goal')
    relative, digest = GOALS[name]
    path = ROOT / relative
    evidence = binding(path)
    require(evidence['sha256'] == digest, 'Committed continuation goal changed')
    value = strict_json(path.read_bytes())
    require(value.get('schema') in (FAILED_ANCHOR_SCHEMA, COMPLETED_ANCHOR_SCHEMA)
            and value.get('id') == name, 'Continuation goal identity changed')
    require(type(value.get('additional_games')) is int and value['additional_games'] > 0
            and value['additional_games'] % 4 == 0,
            'Continuation goal requires a positive multiple of four games')
    require(binding(path) == evidence, 'Continuation goal changed during validation')
    return value, evidence


def completed_anchor(goal, anchor, plan, report, root, bound, continuation):
    """Bind a cleanly completed session, full-state descriptor and consumed anchor."""
    from experiments.headless.contextual_evidence import strict_json
    from experiments.learner.builtin_training_resume import require
    from tools.run_local_training import final_parent
    evidence = {name: bound(goal['completed_run'][name]) for name in
                ('session.json', 'campaign-report.json', 'shutdown.json')}
    run = Path(evidence['session.json']['path']).parent
    require(all(Path(value['path']) == run / name for name, value in evidence.items()),
            'Completed continuation evidence belongs to different runs')
    session, campaign, shutdown = [strict_json(Path(evidence[name]['path']).read_bytes()) for name in
                                   ('session.json', 'campaign-report.json', 'shutdown.json')]
    require(Path(session['run_dir']).resolve() == run
            and Path(session['run_root']).resolve() == root
            and session.get('profile') == campaign.get('profile') == 'original'
            and plan.get('local_session') == evidence['session.json']
            and final_parent(run) == anchor == campaign.get('latest_parent'),
            'Completed continuation anchor differs from the finalized run')
    require(campaign.get('status') == 'finished'
            and campaign.get('stop_reason') == 'training_game_target_reached'
            and campaign.get('error') is None and campaign.get('supervisor_error') is None
            and shutdown.get('supervisor_error') is None
            and campaign.get('latest_progress') == goal['anchor_progress']
                == campaign['full_final_restore'].get('progress') == report.get('cumulative_progress')
            and report.get('optimizer_updates') == 1
            and report.get('update_status') == 'checkpoint_verified'
            and plan.get('batch_sequence') == goal['anchor_sequence']
            and goal['minimum_next_sequence'] == goal['anchor_sequence'] + 1,
            'Completed continuation anchor has incomplete progress or shutdown')
    rows = [row for row in campaign.get('batches', []) if row.get('mode') == 'training']
    require(rows and all(row.get('status') == 'passed' and row.get('completed_games') == 4
                        and row.get('checkpoint_verified') is True for row in rows)
            and len(rows) * 4 == campaign.get('target_games') == session.get('target_games')
            and rows[-1].get('sequence') == goal['anchor_sequence']
            and rows[-1].get('plan') == anchor['plan'] and rows[-1].get('report') == anchor['report']
            and campaign.get('original_continuation') == session.get('original_continuation')
            and campaign['original_continuation'].get('origin_run') == str(root / goal['origin_run'])
            and campaign['original_continuation'].get('admitted_before_session', -1)
                + len(rows) * 4 == goal['admitted_original_at_anchor']
            and goal['admitted_original_at_anchor'] + goal['additional_games'] == 5000,
            'Completed continuation goal differs from the finalized admitted target')
    ledger = continuation['seed_ledger_snapshot']
    sequences = [row['sequence'] for row in ledger['consumed']]
    require(all(type(sequence) is int for sequence in sequences)
            and len(sequences) == len(set(sequences))
            and continuation['next_sequence'] == ledger['next_sequence']
            and ledger['next_sequence'] >= goal['minimum_next_sequence']
            and goal['anchor_sequence'] in sequences and max(sequences) < ledger['next_sequence'],
            'Completed continuation would reuse a consumed reservation')
    # Check after all reads as well: even a valid document cannot change mid-audit.
    for name, value in evidence.items():
        require(bound(goal['completed_run'][name]) == value,
                'Completed continuation evidence changed during validation')
    return evidence


def remaining_games(name, config, parent, original_remaining, continuation):
    """Require the pinned checkpoint in the independently audited current chain."""
    from experiments.headless.contextual_evidence import strict_json
    from experiments.learner import local_script_training as adapter
    from experiments.learner.builtin_training_resume import binding, require
    from tools.run_local_training import verified_batch
    goal, goal_binding = load_goal(name)
    root = Path(config['run_root']).resolve()
    require(config['default_profile'] == 'original'
            and root == (Path(config['artifact_root']).resolve() / goal['run_root_relative'])
            and continuation.get('origin_run') == str(root / goal['origin_run']),
            'Continuation goal belongs to a different official campaign root')

    def bound(relative):
        path = (root / relative['path']).resolve()
        require(path.is_relative_to(root), 'Continuation goal artifact escapes its campaign root')
        result = {**relative, 'path': str(path)}
        require(binding(path) == result, 'Pinned continuation goal evidence changed')
        return result

    anchor = {key: bound(value) for key, value in goal['anchor_parent'].items()}
    anchor_plan, anchor_report = adapter.read_descriptor(anchor)
    require(anchor_plan.get('profile') == 'original' and anchor_plan.get('mode') == 'training'
            and verified_batch(anchor_report, 0), 'Continuation goal anchor is not fully admitted')
    if goal['schema'] == COMPLETED_ANCHOR_SCHEMA:
        completed = completed_anchor(goal, anchor, anchor_plan, anchor_report, root, bound, continuation)
    else:
        failed_plan = bound(goal['failed_plan'])
        failed_report = bound(goal['failed_report'])
        failed_worker = strict_json(Path(bound(goal['failed_worker'])['path']).read_bytes())
        require(failed_worker.get('plan_sha256') == failed_plan['sha256']
                and failed_worker.get('training_performed') is False
                and failed_worker.get('optimizer_updates') == failed_worker.get('optimizer_steps')
                    == failed_worker.get('training_samples') == 0
                and failed_worker.get('update_status') == 'not_started'
                and len(failed_worker.get('attempts', [])) == 4
                and all(row.get('completion_receipt') is False and row.get('batch_admission') == 'rejected'
                        for row in failed_worker['attempts']),
                'Failed continuation batch is no longer fully excluded')
        ledger = continuation['seed_ledger_snapshot']
        require(continuation['next_sequence'] == ledger['next_sequence']
                and ledger['next_sequence'] >= goal['minimum_next_sequence']
                and goal['failed_sequence'] in {row['sequence'] for row in ledger['consumed']},
                'Continuation would reuse a consumed failed reservation')
    cursor, seen, updates = parent, set(), 0
    while cursor != anchor:
        require(type(cursor) is dict and set(cursor) == adapter.DESCRIPTOR_KEYS
                and cursor['plan']['sha256'] not in seen and updates < goal['additional_games'] // 4,
                'Latest checkpoint does not descend from the goal anchor within its game budget')
        seen.add(cursor['plan']['sha256'])
        plan, report = adapter.read_descriptor(cursor)
        require(plan.get('profile') == 'original' and plan.get('mode') == 'training'
                and plan.get('scope') == adapter.SCOPE and verified_batch(report, 0)
                and report.get('optimizer_updates') == 1
                and report.get('update_status') == 'checkpoint_verified',
                'Continuation goal descendant is not a fully admitted four-game update')
        if goal['schema'] == COMPLETED_ANCHOR_SCHEMA:
            require(plan.get('policy_support') == goal['required_policy_support'],
                    'Continuation goal descendant uses a different policy support intervention')
        session = strict_json(Path(plan['local_session']['path']).read_bytes())
        require(binding(plan['local_session']['path']) == plan['local_session']
                and Path(session['run_root']).resolve() == root,
                'Continuation goal descendant belongs to another campaign')
        cursor = plan['parent_descriptor']
        updates += 1
    admitted = 4 * updates
    require(continuation['admitted_before_session'] == goal['admitted_original_at_anchor'] + admitted
            and continuation['remaining_before_session'] == original_remaining
            and continuation['target_games'] == 5000
            and original_remaining == 5000 - continuation['admitted_before_session'],
            'Continuation goal count differs from independently verified original admissions')
    remaining = goal['additional_games'] - admitted
    require(0 <= remaining <= original_remaining, 'Continuation goal exceeds the original remaining target')
    audit = dict(schema='heroes-manual-continuation-goal-audit-v1', goal=goal_binding,
        goal_id=name, anchor_progress=goal['anchor_progress'], additional_games=goal['additional_games'],
        admitted_since_anchor=admitted, remaining_games=remaining,
        next_sequence=continuation['next_sequence'], anchor_parent=anchor)
    if goal['schema'] == COMPLETED_ANCHOR_SCHEMA:
        audit.update(schema='heroes-completed-run-continuation-goal-audit-v1',
                     completed_run=completed, required_policy_support=goal['required_policy_support'])
    else:
        audit.update(failed_plan=failed_plan, failed_report=failed_report)
    return remaining, audit
