#pragma once

// Read-only instrumentation for a separately built diagnostic replay binary.
// This header is not part of the accepted engine/runtime recipe. Full inventory
// output is private forensic evidence and must never enter a policy observation.
#include "lib/StartInfo.h"
#include "lib/entities/artifact/ArtifactUtils.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/artifact/CArtifactInstance.h"
#include "lib/gameState/CGameState.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapping/CMap.h"
#include "lib/networkPacks/PacksForServer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace artifact_inventory_audit
{
inline std::string line(const JsonNode & value)
{
	auto encoded = value.toCompactString();
	encoded.erase(std::remove_if(encoded.begin(), encoded.end(),
		[](char c) { return c == '\n' || c == '\r'; }), encoded.end());
	return encoded;
}

inline JsonNode artifact(const CArtifactInstance * item)
{
	JsonNode result;
	result["present"].Bool() = item != nullptr;
	if(item)
	{
		result["instance_id"].Integer() = item->getId().getNum();
		result["type_id"].Integer() = item->getTypeId().getNum();
		const auto * type = item->getType();
		result["type_present"].Bool() = type != nullptr;
		if(type)
		{
			result["json_key"].String() = type->getJsonKey();
			result["name"].String() = type->getNameTranslated();
		}
	}
	return result;
}

inline JsonNode slot(const CArtifactSet & inventory, ArtifactPosition position)
{
	JsonNode result;
	result["slot"].Integer() = position.getNum();
	result["position_valid"].Bool() = ArtifactUtils::checkIfSlotValid(inventory, position);
	const auto * entry = inventory.getSlot(position);
	result["slot_entry_present"].Bool() = entry != nullptr;
	result["locked"].Bool() = entry && entry->locked;
	result["artifact"] = artifact(entry ? entry->getArt() : nullptr);
	return result;
}

inline JsonNode bonuses(const StartInfo * info)
{
	JsonNode result;
	result.Vector();
	if(!info) return result;
	for(const auto & [player, settings] : info->playerInfos)
	{
		JsonNode entry;
		entry["player"].Integer() = player.getNum();
		entry["bonus"].Integer() = static_cast<int>(settings.bonus);
		switch(settings.bonus)
		{
			case PlayerStartingBonus::RANDOM: entry["bonus_name"].String() = "random"; break;
			case PlayerStartingBonus::ARTIFACT: entry["bonus_name"].String() = "artifact"; break;
			case PlayerStartingBonus::GOLD: entry["bonus_name"].String() = "gold"; break;
			case PlayerStartingBonus::RESOURCE: entry["bonus_name"].String() = "resource"; break;
			default: entry["bonus_name"].String() = "unknown"; break;
		}
		result.Vector().push_back(std::move(entry));
	}
	return result;
}

inline JsonNode inventories(const CGameState & state)
{
	JsonNode result;
	result.Vector();
	for(const auto * object : state.getMap().getObjects())
	{
		const auto * hero = dynamic_cast<const CGHeroInstance *>(object);
		if(!hero) continue;
		JsonNode entry;
		entry["hero_id"].Integer() = hero->id.getNum();
		entry["hero_name"].String() = hero->getNameTranslated();
		entry["owner"].Integer() = hero->getOwner().getNum();
		auto & worn = entry["worn"].Vector();
		for(const auto & [position, ignored] : hero->artifactsWorn)
			worn.push_back(slot(*hero, position));
		auto & backpack = entry["backpack"].Vector();
		for(size_t index = 0; index < hero->artifactsInBackpack.size(); ++index)
			backpack.push_back(slot(*hero, ArtifactPosition(ArtifactPosition::BACKPACK_START + index)));
		entry["transition"] = slot(*hero, ArtifactPosition::TRANSITION_POS);
		result.Vector().push_back(std::move(entry));
	}
	return result;
}

inline JsonNode location(const CGameState & state, const ArtifactLocation & value)
{
	JsonNode result;
	result["holder_id"].Integer() = value.artHolder.getNum();
	result["slot"].Integer() = value.slot.getNum();
	result["creature"] = value.creature ? JsonNode(int64_t(value.creature->getNum())) : JsonNode();
	const auto * holder = state.getObj(value.artHolder);
	const auto * hero = state.getHero(value.artHolder);
	result["holder_present"].Bool() = holder != nullptr;
	result["hero_present"].Bool() = hero != nullptr;
	result["source_present"] = JsonNode();
	result["inspection_status"].String() = "unsupported_location";
	if(!holder)
	{
		result["source_present"].Bool() = false;
		result["inspection_status"].String() = "missing_holder";
	}
	// Keep unsupported location kinds explicit instead of treating them as empty.
	// In particular, getArtSet has non-hero branches requiring a creature value.
	result["inspection_supported"].Bool() = hero && !value.creature;
	if(hero)
	{
		result["hero_name"].String() = hero->getNameTranslated();
		result["owner"].Integer() = hero->getOwner().getNum();
		if(!value.creature)
		{
			result["contents"] = slot(*hero, value.slot);
			result["source_present"] = result["contents"]["artifact"]["present"];
			result["inspection_status"].String() = result["source_present"].Bool() ? "artifact_present" : "empty_slot";
		}
	}
	return result;
}

class Audit
{
	std::ofstream output;
	std::string previousInventory;
	uint64_t inventoryChanges = 0;
	uint64_t exchangeRequests = 0;

	void write(JsonNode value)
	{
		value["schema"].String() = "heroes-artifact-inventory-audit-v1";
		value["diagnostic_only"].Bool() = true;
		output << line(value) << '\n';
		output.flush();
		if(!output) throw std::runtime_error("Artifact inventory audit write failed");
	}

public:
	Audit()
	{
		const auto * destination = std::getenv("HEROES_ARTIFACT_INVENTORY_AUDIT");
		if(!destination || !*destination || !std::filesystem::path(destination).is_absolute()
			|| std::filesystem::exists(destination))
			throw std::runtime_error("Artifact audit requires a fresh absolute private output path");
		output.open(destination, std::ios::out);
		if(!output) throw std::runtime_error("Cannot create artifact inventory audit");
	}

	void initialize(const CGameState & state)
	{
		JsonNode value;
		value["kind"].String() = "initial_state";
		value["initial_starting_bonuses"] = bonuses(state.getInitialStartInfo());
		value["resolved_starting_bonuses"] = bonuses(state.getStartInfo());
		value["inventories"] = inventories(state);
		previousInventory = line(value["inventories"]);
		write(std::move(value));
		// Isolated read-only negative controls are not recorded game requests.
		JsonNode controls;
		controls["kind"].String() = "diagnostic_fixture";
		controls["purpose"].String() = "missing hero and empty slot inspection";
		controls["missing_hero"] = location(state, ArtifactLocation(ObjectInstanceID(999999), ArtifactPosition(9999)));
		if(controls["missing_hero"]["hero_present"].Bool()
			|| controls["missing_hero"]["source_present"].Bool())
			throw std::runtime_error("Missing hero audit fixture unexpectedly exists");
		bool checkedEmpty = false;
		for(const auto * object : state.getMap().getObjects())
		{
			const auto * hero = dynamic_cast<const CGHeroInstance *>(object);
			if(!hero) continue;
			const auto position = ArtifactPosition(ArtifactPosition::BACKPACK_START + hero->artifactsInBackpack.size() + 100);
			controls["empty_slot"] = location(state, ArtifactLocation(hero->id, position));
			if(controls["empty_slot"]["source_present"].Bool())
				throw std::runtime_error("Empty slot audit fixture unexpectedly contains an artifact");
			checkedEmpty = true;
			break;
		}
		controls["empty_slot_checked"].Bool() = checkedEmpty;
		write(std::move(controls));
	}

	void afterPacket(const CGameState & state, uint64_t packetIndex, int packetType)
	{
		auto current = inventories(state);
		const auto encoded = line(current);
		if(encoded == previousInventory) return;
		JsonNode value;
		value["kind"].String() = "inventory_change";
		value["client_packet_index"].Integer() = packetIndex;
		value["client_packet_type"].Integer() = packetType;
		value["inventories"] = std::move(current);
		previousInventory = encoded;
		++inventoryChanges;
		write(std::move(value));
	}

	void request(const CGameState & state, const CPackForServer & request,
		const JsonNode & record, const JsonNode & accounting)
	{
		const auto * exchange = dynamic_cast<const ExchangeArtifacts *>(&request);
		if(!exchange) return;
		JsonNode value;
		value["kind"].String() = "exchange_artifacts_request";
		value["request"] = record;
		value["source"] = location(state, exchange->src);
		value["destination"] = location(state, exchange->dst);
		value["state_boundary"].String() = "regenerated_client_state_at_recorded_send";
		value["semantic_scope"].String() = "artifact presence and slot compatibility; not full server authorization";
		value["request_applied"] = JsonNode();
		value["application_status"].String() = "not_classified_by_inventory_audit";
		for(const auto & cancelled : accounting["terminal_unapplied"].Vector())
		{
			bool same = true;
			for(const auto * key : {"seq", "player", "request_id", "pack_type", "pack", "source"})
				same = same && cancelled[key] == record[key];
			if(same)
			{
				value["request_applied"].Bool() = false;
				value["application_status"].String() = "verified_terminal_unapplied";
				value["receipt"] = cancelled["receipt"];
			}
		}
		const auto * source = state.getHero(exchange->src.artHolder);
		const auto * destination = state.getHero(exchange->dst.artHolder);
		value["source_present"] = value["source"]["source_present"];
		value["destination_compatible"] = JsonNode();
		if(source && !exchange->src.creature)
		{
			const auto * entry = source->getSlot(exchange->src.slot);
			const auto * item = entry ? entry->getArt() : nullptr;
			value["source_present"].Bool() = item != nullptr;
			if(item && destination && !exchange->dst.creature
				&& ArtifactUtils::checkIfSlotValid(*destination, exchange->dst.slot))
				value["destination_compatible"].Bool() = item->canBePutAt(destination, exchange->dst.slot, true);
		}
		++exchangeRequests;
		write(std::move(value));
	}

	void complete()
	{
		JsonNode value;
		value["kind"].String() = "complete";
		value["inventory_changes"].Integer() = inventoryChanges;
		value["exchange_requests"].Integer() = exchangeRequests;
		value["official_training_samples"].Integer() = 0;
		write(std::move(value));
	}
};
}
