"""Diagnostic v6 mutations selected only from fields actually recorded.

These alter copies for the independent native verifier. They are neither
rollouts nor learner inputs, and never create missing mechanic coverage.
"""
from experiments.learner import contextual as contract


def army_management_mutations(tape, add):
    if not tape:
        return
    profile = contract.MANAGEMENT_PROFILE
    header = tape[0]
    if (header.get("schema") != "heroes-contextual-replay-v1"
            or header.get("interface") != profile.environment
            or type(header.get("protocol_version")) is not int
            or header.get("protocol_version") != profile.protocol
            or header.get("history_schema") != profile.history_schema
            or header.get("action_schema") != profile.action_schema):
        return
    selected = set()
    observed = {}

    def change(name, index, path, value):
        if name in selected:
            return
        selected.add(name)
        def mutate(rows, i):
            target = rows[i]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
        add(name, index, mutate)

    for index, row in enumerate(tape):
        player = row.get("player")
        if row.get("kind") == "message":
            for r, record in enumerate(row.get("message", {}).get("records", [])):
                prefix = ("message", "records", r)
                obs = record.get("observation")
                if obs is not None:
                    observed[player] = obs
                    for h, hero in enumerate(obs.get("heroes", [])):
                        if hero.get("in_garrison") is True:
                            change("garrisoned_hero_flag", index, (*prefix, "observation", "heroes", h, "in_garrison"), False)
                    window = obs.get("window") or {}
                    for a, army in enumerate(window.get("armies", [])):
                        for s, slot in enumerate(army["slots"]):
                            if slot["count"]:
                                change("displayed_army_count", index,
                                       (*prefix, "observation", "window", "armies", a, "slots", s, "count"), slot["count"] + 1)
                    panel = window.get("stack_panel")
                    if panel:
                        base = (*prefix, "observation", "window", "stack_panel")
                        if panel["offers"]:
                            change("displayed_upgrade_creature", index, (*base, "offers", 0, "creature"), panel["offers"][0]["creature"] + 1)
                        if panel["selected_upgrade_ref"] is not None:
                            change("selected_upgrade_reference", index, (*base, "selected_upgrade_ref"), panel["selected_upgrade_ref"] + "-changed")
                            change("upgrade_native_cost", index, (*base, "total_cost", 6), panel["total_cost"][6] + 1)
                            change("upgrade_affordability", index, (*base, "affordable"), not panel["affordable"])
                if record.get("event", {}).get("kind") == "army_exchange_requested":
                    change("exchange_requested_partner", index, (*prefix, "event", "partner"), record["event"]["subject"])
                if record.get("kind") == "context_opened" and record.get("reason") == "own_hero_exchange":
                    change("exchange_opened_partner", index, (*prefix, "context", "selection"), record["context"]["subject"])
                    change("exchange_opened_revision", index, (*prefix, "revision"), record["revision"] + 1)
        if row.get("kind") != "response":
            continue
        action = row["response"]["action"]
        kind = action["kind"]
        prefix = ("response", "action")
        if kind in ("split_stack", "merge_stacks", "swap_stacks"):
            obs = observed.get(player) or {}
            slots = [slot for army in (obs.get("window") or {}).get("armies", []) for slot in army["slots"]]
            alternative = next((slot["ref"] for slot in slots if slot["ref"] not in (action["selection"], action["destination"])), None)
            if alternative is not None:
                change(kind + "_source", index, (*prefix, "selection"), alternative)
                change(kind + "_destination", index, (*prefix, "destination"), alternative)
            if kind == "split_stack":
                change("exact_split_count", index, (*prefix, "count"), action["count"] - 1 if action["count"] > 1 else 2)
        if kind == "swap_garrison":
            change("garrison_swap_subject", index, (*prefix, "subject"), action["subject"] + "-changed")
        if kind in ("select_upgrade", "upgrade_stack"):
            change(kind + "_selection", index, (*prefix, "selection"), action["selection"] + "-changed")
        if kind == "close" and (observed.get(player, {}).get("window") or {}).get("context", {}).get("kind") == "exchange":
            change("exchange_close_action", index, (*prefix, "kind"), "end_turn")
