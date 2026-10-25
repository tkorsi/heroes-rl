"""V7 replay negatives selected solely from recorded player-visible fields.

The caller must verify the untouched native replay before running these copies.
No missing query family, displayed component or gameplay outcome is synthesized.
"""
from copy import deepcopy

from experiments.learner import contextual as contract


def adventure_dialogs_mutations(tape, add):
    if not tape:
        return
    profile = contract.DIALOGS_PROFILE
    header = tape[0]
    expected = {"schema": "heroes-contextual-replay-v1", "interface": profile.environment,
                "protocol_version": profile.protocol, "history_schema": profile.history_schema,
                "action_schema": profile.action_schema}
    if (any(header.get(key) != value for key, value in expected.items())
            or type(header.get("protocol_version")) is not int):
        return
    selected, observed, answered, previous_choices = set(), {}, set(), {}

    def change(name, index, path, value):
        if name in selected:
            return
        selected.add(name)
        def mutate(rows, i):
            target = rows[i]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = deepcopy(value)
        add(name, index, mutate)

    def displayed_panel(panel, index, base, label):
        prefix = "dialog_" + label + "_"
        change(prefix + "reference", index, (*base, "ref"), panel["ref"] + "-changed")
        change(prefix + "family", index, (*base, "family"), "guarded_mine" if panel["family"] != "guarded_mine" else "neutral_free_join")
        change(prefix + "selection_flag", index, (*base, "selection"), not panel["selection"])
        change(prefix + "cancel_flag", index, (*base, "cancel"), not panel["cancel"])
        first = panel["text"][0]["text_id"]
        change(prefix + "text", index, (*base, "text", 0, "text_id"), "core.advevent.84" if first != "core.advevent.84" else "core.advevent.187")
        for t, token in enumerate(panel["text"]):
            if token.get("number") is not None:
                change(prefix + "number", index, (*base, "text", t, "number"), token["number"] + 1)
            if token.get("op") == "replace_text" and (token.get("text_id") or "").startswith("creatures.core."):
                name = token["text_id"]
                changed = name.rsplit(".", 1)[0] + (".singular" if name.endswith(".plural") else ".plural")
                change(prefix + "creature_name", index, (*base, "text", t, "text_id"), changed)
        for component_index, component in enumerate(panel["components"]):
            change(prefix + "component_amount", index, (*base, "components", component_index, "amount"), component["amount"] + 1)
            if component.get("skill") is not None:
                change(prefix + "component_skill", index, (*base, "components", component_index, "skill"),
                       "attack" if component["skill"] != "attack" else "defence")
        for choice_index, choice in enumerate(panel["choices"]):
            change(prefix + "choice_reference", index, (*base, "choices", choice_index, "ref"), choice["ref"] + "-changed")
            change(prefix + "choice_kind", index, (*base, "choices", choice_index, "kind"),
                   "decline" if choice["kind"] != "decline" else "accept")
            if choice.get("component_index") is not None:
                change(prefix + "choice_component", index, (*base, "choices", choice_index, "component_index"), 1 - choice["component_index"])

    for index, row in enumerate(tape):
        player = row.get("player")
        if row.get("kind") == "message":
            for r, record in enumerate(row.get("message", {}).get("records", [])):
                base = ("message", "records", r)
                context = record.get("context")
                if context is not None and context.get("kind") != "dialog":
                    observed.pop(player, None)
                obs = record.get("observation")
                if obs is not None:
                    observed[player] = obs
                    window = obs.get("window") or {}
                    panel = window.get("dialog_panel")
                    if panel is not None and window.get("context", {}).get("kind") == "dialog":
                        displayed_panel(panel, index, (*base, "observation", "window", "dialog_panel"), "view")
                        change("dialog_view_context_reference", index, (*base, "observation", "window", "context", "selection"), panel["ref"] + "-changed")
                event = record.get("event") or {}
                if event.get("kind") == "dialog_requested":
                    displayed_panel(event["dialog"], index, (*base, "event", "dialog"), "callback")
                    change("dialog_callback_subject", index, (*base, "event", "subject"), event["subject"] + "-changed")
                if record.get("kind") == "context_opened" and record.get("reason") == "adventure_dialog":
                    change("dialog_opened_reference", index, (*base, "context", "selection"), context["selection"] + "-changed")
                    change("dialog_opened_subject", index, (*base, "context", "subject"), context["subject"] + "-changed")
                    change("dialog_opened_revision", index, (*base, "revision"), record["revision"] + 1)
                    change("dialog_opened_reason", index, (*base, "reason"), "own_hero_exchange")
                decision = record.get("decision_id")
                if (player, decision) in answered:
                    if record.get("kind") == "acknowledged":
                        change("dialog_reply_acknowledgement", index, (*base, "accepted"), not record["accepted"])
                    if record.get("kind") == "result":
                        if record["result"].get("status") == "dialog_answered":
                            change("dialog_reply_result", index, (*base, "result", "status"), "closed")
                            change("dialog_reply_acceptance", index, (*base, "result", "accepted"), False)
                        answered.discard((player, decision))
                    if record.get("kind") == "superseded":
                        answered.discard((player, decision))
                notice = event.get("notice")
                if notice is not None and notice.get("text"):
                    first = notice["text"][0].get("text_id")
                    if first in contract.DIALOG_NOTICE_TEXT_IDS:
                        change("dialog_notice_mode", index, (*base, "event", "notice", "display_mode"),
                               "modal" if notice["display_mode"] != "modal" else "auto")
                    if first in ("core.advevent.115", "core.genrltxt.139", "core.advevent.145"):
                        for component_index, component in enumerate(notice.get("components", [])):
                            target = (*base, "event", "notice", "components", component_index)
                            if component.get("amount") is not None:
                                label = "dialog_scholar_mastery" if first == "core.genrltxt.139" else "dialog_map_scholar_amount"
                                amount = component["amount"]
                                change(label, index, (*target, "amount"), 1 if amount == 3 else amount + 1)
                            if component.get("content_id") is not None:
                                label = ("dialog_chest_artifact" if first == "core.advevent.145" else
                                         "dialog_scholar_spell" if component["kind"] == "spell" else "dialog_scholar_skill")
                                change(label, index, (*target, "content_id"), component["content_id"] + "Changed")
                            if component.get("skill") is not None:
                                change("dialog_map_scholar_primary", index, (*target, "skill"),
                                       "attack" if component["skill"] != "attack" else "defence")
                        for token_index, token in enumerate(notice["text"]):
                            if token.get("text_id") == "heroes.notice.comma_space":
                                change("dialog_scholar_separator", index, (*base, "event", "notice", "text", token_index, "text_id"), "core.genrltxt.141")
        if row.get("kind") != "response":
            continue
        response = row["response"]
        action = response["action"]
        if action.get("kind") != "choose_dialog":
            continue
        obs = observed.get(player) or {}
        window = obs.get("window") or {}
        panel = window.get("dialog_panel")
        if (panel is None or window.get("context", {}).get("kind") != "dialog"
                or action.get("subject") != window["context"].get("subject")):
            continue
        choices = [choice["ref"] for choice in panel["choices"]]
        if action.get("selection") not in choices:
            continue
        alternative = next(ref for ref in choices if ref != action["selection"])
        change("dialog_answer_choice", index, ("response", "action", "selection"), alternative)
        own = next((hero["handle"] for hero in obs["heroes"] if hero["handle"] != action["subject"]), action["subject"] + "-changed")
        change("dialog_answer_subject", index, ("response", "action", "subject"), own)
        stale = next((ref for ref in previous_choices.get(player, ()) if ref not in choices), None)
        if stale is not None:
            change("dialog_answer_stale_choice", index, ("response", "action", "selection"), stale)
        previous_choices[player] = choices
        if type(response.get("decision_id")) is int:
            answered.add((player, response["decision_id"]))
