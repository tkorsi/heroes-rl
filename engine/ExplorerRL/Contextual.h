#pragma once

#include "../../lib/json/JsonNode.h"
#include "TownEconomy.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace explorer_rl
{
/// What the tiles and objects of an observation are written from, as plain
/// numbers. Two of them are equal exactly when the JSON written from them is,
/// so a capture that changed nothing on the map is found without building the
/// JSON of every visible tile.
struct VisibleMap
{
	struct Tile
	{
		int x, y, terrain, road;
		bool blocked, visitable;
		bool operator==(const Tile &) const = default;
	};
	struct Object
	{
		int id, x, y, kind, subtype;
		int owner; ///< 0 self, 1 opponent, 2 neutral: the relative owner the policy is told
		bool operator==(const Object &) const = default;
	};
	std::vector<Tile> tiles; ///< row by row, as the observation lists them
	std::vector<Object> objects; ///< in the order their tiles were met
	bool operator==(const VisibleMap &) const = default;
};

/// One stack's entry of an "own_armies" army, kept for as long as nothing it
/// was read from has changed. Every figure in it but the creature and the
/// count is an answer of the stack's bonus node, and the engine keeps those
/// answers itself for as long as the node's tree version stands
/// (CBonusSystemNode::getAllBonuses). The same stack in the same slot with
/// the same creature, count and version therefore has the entry it had.
struct OwnStackDetail
{
	int32_t version = -1; ///< the stack's bonus tree version, read before the entry was built
	int slot = -1;
	int creature = -1;
	int64_t count = -1;
	JsonNode item; ///< null until built
};
using OwnStackDetails = std::unordered_map<const void *, OwnStackDetail>; ///< by the stack's address

/// What a view in the map context is written from besides its tiles and
/// objects: the player's heroes and towns with their places, movement and
/// stacks (as OwnStackDetail knows a stack), and the resources. Two views in
/// the map context with equal prints and equal maps are the same JSON, so a
/// capture that finds the print and the map of the latest view again need not
/// build its observation to learn that nothing changed.
struct ViewPrint
{
	std::vector<int64_t> numbers;
	std::vector<const void *> stacks;
	bool operator==(const ViewPrint &) const = default;
};

/// Everything a capture's observation can depend on besides the game state, with the game state's
/// generation: a capture whose signature equals the one of the capture before it would build the
/// same observation (see ExplorerRL::captureSignature).
struct CaptureSignature
{
	uint64_t generation = 0; ///< packs applied that can change the game state
	int day = 0;
	std::string context, subject, panel, selection, selectedUpgradeRef, capabilityRef;
	std::vector<size_t> sizes; ///< of the routing tables, which only grow
	size_t selectedTavernIndex = 0, selectedRecruitIndex = 0;
	std::map<int, std::string> tavernHandles, upgradeHandles;
	JsonNode stackParentContext, capabilityParentContext;
	std::optional<int> capabilityInspected;
	std::vector<int64_t> exchange; ///< ownHeroExchange's fields, empty without one
	std::vector<int64_t> dialogs; ///< adventure and capability dialogs' heroes, queries, installed flags
	JsonNode adventurePanel, capabilityPanel;
	std::map<std::string, int> adventureReplies, capabilityReplies;
	bool operator==(const CaptureSignature &) const = default;
};

/// Per-player adapter bookkeeping. All access is protected by ExplorerRL's
/// state mutex; engine identifiers never leave these routing tables.
struct ContextualState
{
	std::map<int, std::string> handles;
	std::map<std::string, int> subjects;
	std::vector<JsonNode> records;
	JsonNode lastObservation; ///< of the latest view, without its tiles and objects
	VisibleMap lastMap; ///< the tiles and objects of the latest view
	OwnStackDetails ownStackDetails; ///< entries of the "own_armies" stacks seen last; a miss only builds one again
	/// The print of lastObservation while that was captured in the map context
	/// and no action has run since (an action reads the latest view and may
	/// leave a mark in it); otherwise empty, and the next capture compares in full.
	std::optional<ViewPrint> lastPrint;
	/// The signature of the latest capture, while nothing that the signature leaves out can have
	/// changed lastObservation since (reset where lastPrint is reset); otherwise empty.
	std::optional<CaptureSignature> lastSignature;
	/// The shape (node types and sizes) of lastObservation when it was last known to be as a capture left it.
	/// Reading it through JsonNode's non-const accessors can only insert members or change a node's type,
	/// both of which change the shape; an action reads it so (see executeContextualAction).
	std::optional<uint64_t> lastShape;
	int lastDay = 0;
	std::string context = "map";
	std::string subject;
	std::string panel; ///< empty outside the opt-in v3 town context
	std::string selection;
	TownEconomyState economy;
	std::map<int, std::string> tavernHandles; ///< private offered HeroTypeID routing
	size_t selectedTavernIndex = 0;
	std::map<std::pair<int, int>, std::string> armyHandles;
	std::map<std::tuple<int, int, int>, std::string> armySlotHandles;
	JsonNode stackParentContext; ///< exact opened parent; never refreshed while stack is open
	std::string selectedUpgradeRef;
	std::map<int, std::string> upgradeHandles; ///< private CreatureID routing
	struct OwnHeroExchange
	{
		int first;
		int second;
		int query;
		bool installed = false;
		bool joiningNeutral = false;
		bool objectGarrison = false;
		bool removableUnits = true;
		bool allied = false;
	};
	std::optional<OwnHeroExchange> ownHeroExchange; ///< private native query, never policy input
	struct AdventureDialog
	{
		int hero;
		int query;
		JsonNode panel;
		std::map<std::string, int> replies;
		bool installed = false;
	};
	std::optional<AdventureDialog> adventureDialog; ///< private query/reply routing
	struct CapabilityDialog
	{
		int hero;
		int query;
		JsonNode panel;
		std::map<std::string, int> replies;
		bool installed = false;
	};
	std::optional<CapabilityDialog> capabilityDialog;
	JsonNode capabilityParentContext;
	std::string capabilityRef;
	std::optional<int> capabilityInspected;
	uint64_t nextCapability = 0;
	std::optional<int> executingHero; ///< own subject of current move or dialog answer
	uint64_t nextDialog = 0;
	int64_t revision = -1;
	int64_t nextDecision = 0;
	std::optional<int64_t> decision;
	bool executing = false;
	bool issueArmed = false;
	int64_t commands = 0;
	int64_t interactions = 0;
	std::optional<int64_t> routeInteractions;
	struct Request { int64_t decision; int64_t command; };
	std::map<int, Request> requests;

	std::string handle(int engineID);
	JsonNode contextJson() const;
	JsonNode record(const char * kind, int day) const;
	void supersede(int day, const char * reason);
};
}
