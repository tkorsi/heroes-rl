#include "StdInc.h"
#include "KnowledgePlanner.h"
#include "VisibleWorld.h"

#include "../../lib/CPlayerState.h"
#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/ObjectTemplate.h"
#include "../../lib/mapping/MapTilesStorage.h"
#include "../../lib/mapping/TerrainTile.h"
#include "../../lib/pathfinder/CGPathNode.h"
#include "../../lib/pathfinder/CPathfinder.h"
#include "../../lib/pathfinder/NodeStorage.h"
#include "../../lib/pathfinder/PathfinderOptions.h"
#include "../../lib/pathfinder/PathfindingRules.h"
#include "../../lib/pathfinder/PathfinderUtil.h"
#include "../../lib/pathfinder/TurnInfo.h"

#include <unordered_map>

// The pathfinder reads the world only through the IGameInfoCallback it is
// given, the TerrainTile objects that callback returns, the objects it hands
// out, and the hero's own state. KnowledgeMap answers every such call from one
// player's knowledge, and the rules below replace the stock rule that asks a
// town or garrison whether its hidden army lets the hero pass. This follows
// the VCMI 1.7.5 sources; a new engine version must be audited again.
namespace explorer_rl
{
namespace
{
[[noreturn]] void outsideKnowledge(const char * method)
{
	throw std::logic_error(std::string("Knowledge routing does not provide ") + method);
}

GridPosition gridPosition(const int3 & position)
{
	return {position.x, position.y, position.z};
}

/// ObjectTemplate::isVisitableFrom for each neighbouring offset, as KnownObject::visitDir bits.
uint8_t visitDirections(const CGObjectInstance & object)
{
	static constexpr std::array<std::array<int, 3>, 8> directions{{
		{{-1, -1, 1}}, {{0, -1, 2}}, {{1, -1, 4}}, {{1, 0, 8}},
		{{1, 1, 16}}, {{0, 1, 32}}, {{-1, 1, 64}}, {{-1, 0, 128}}
	}};
	uint8_t mask = 0;
	for(const auto & [dx, dy, bit] : directions)
		if(object.appearance->isVisitableFrom(dx, dy))
			mask |= bit;
	return mask;
}

class KnowledgeMap final : public IGameInfoCallback
{
public:
	KnowledgeMap(const CCallback & player, PlayerColor color)
		: player(player), color(color), size(player.getMapSize()), grid(size.x, size.y, size.z), tiles(size), guards(size)
	{
		// Rock is impassable land: an unknown tile is never a neighbour, never
		// holds an object and never guards, so nothing behind the fog is read.
		unknown.terrainType = ETerrainId::ROCK;
		for(int z = 0; z < size.z; ++z)
			for(int y = 0; y < size.y; ++y)
				for(int x = 0; x < size.x; ++x)
					capture(int3(x, y, z));
		for(int z = 0; z < size.z; ++z)
			for(int y = 0; y < size.y; ++y)
				for(int x = 0; x < size.x; ++x)
				{
					const auto guard = grid.guard({x, y, z});
					guards[int3(x, y, z)] = guard ? int3((*guard)[0], (*guard)[1], (*guard)[2]) : int3(-1, -1, -1);
				}
	}

	bool known(const int3 & position) const { return isInTheMap(position) && grid.at(gridPosition(position)).known; }

	/// PathfinderUtil::evaluateAccessibility<LAND> over knowledge.
	EPathAccessibility landAccessibility(const int3 & position) const
	{
		if(!known(position))
			return EPathAccessibility::BLOCKED;
		const auto & tile = tiles[position];
		if(tile.visitable())
		{
			if(tile.visitableObjects.size() > 1)
			{
				const auto * front = getObjInstance(tile.visitableObjects.front());
				const auto * back = getObjInstance(tile.visitableObjects.back());
				if(front->ID == Obj::SANCTUARY && back->ID == Obj::HERO && back->getOwner() != color)
					return EPathAccessibility::BLOCKED;
				return EPathAccessibility::ACCESSIBLE;
			}
			const auto * object = getObjInstance(tile.visitableObjects.front());
			if(object->isBlockedVisitable())
				return EPathAccessibility::BLOCKVIS;
			return knownPassable(*object) ? EPathAccessibility::ACCESSIBLE : EPathAccessibility::VISITABLE;
		}
		if(tile.blocked())
			return EPathAccessibility::BLOCKED;
		if(guards[position].isValid())
			return EPathAccessibility::GUARDED;
		return EPathAccessibility::ACCESSIBLE;
	}

	/// CGObjectInstance::passableFor as far as the player knows. Towns and
	/// garrisons let anyone pass while their garrison is empty, which the player
	/// cannot see, so only its own and allied ones count as passable. A border
	/// gate depends on the player's own keymaster visits, which it does know.
	/// Any other object keeps the base answer: not passable.
	bool knownPassable(const CGObjectInstance & object) const
	{
		if(object.ID == Obj::TOWN || object.ID == Obj::GARRISON || object.ID == Obj::GARRISON2)
			return player.getPlayerRelations(object.getOwner(), color) != PlayerRelations::ENEMIES;
		if(object.ID == Obj::BORDER_GATE)
			return object.passableFor(color);
		return false;
	}

	CGameState & gameState() override { outsideKnowledge("gameState"); }
	const CGameState & gameState() const override { outsideKnowledge("gameState"); }
	int getDate(Date) const override { outsideKnowledge("getDate"); }
	const CMapHeader * getMapHeader() const override { outsideKnowledge("getMapHeader"); }
	const StartInfo * getStartInfo() const override { outsideKnowledge("getStartInfo"); }
	bool isAllowed(SpellID) const override { outsideKnowledge("isAllowed"); }
	bool isAllowed(ArtifactID) const override { outsideKnowledge("isAllowed"); }
	bool isAllowed(SecondarySkill) const override { outsideKnowledge("isAllowed"); }

	bool isVisibleFor(int3 position, PlayerColor other) const override
	{
		if(other != color)
			outsideKnowledge("isVisibleFor another player");
		return known(position);
	}

	bool isVisibleFor(const CGObjectInstance *, PlayerColor) const override { outsideKnowledge("isVisibleFor an object"); }
	const IGameSettings & getSettings() const override { return player.getSettings(); }
	int3 getMapSize() const override { return size; }
	bool isInTheMap(const int3 & position) const override { return player.isInTheMap(position); }
	const TeamState * getTeam(TeamID) const override { outsideKnowledge("getTeam"); }

	const TeamState * getPlayerTeam(PlayerColor other) const override
	{
		if(other != color)
			outsideKnowledge("getPlayerTeam for another player");
		return player.getPlayerTeam(other);
	}

	const PlayerState * getPlayerState(PlayerColor other, bool verbose) const override
	{
		if(other != color)
			outsideKnowledge("getPlayerState for another player");
		return player.getPlayerState(other, verbose);
	}

	const PlayerSettings * getPlayerSettings(PlayerColor) const override { outsideKnowledge("getPlayerSettings"); }
	PlayerRelations getPlayerRelations(PlayerColor first, PlayerColor second) const override { return player.getPlayerRelations(first, second); }
	int getHeroCount(PlayerColor, bool) const override { outsideKnowledge("getHeroCount"); }
	EPlayerStatus getPlayerStatus(PlayerColor, bool) const override { outsideKnowledge("getPlayerStatus"); }
	int getResource(PlayerColor, GameResID) const override { outsideKnowledge("getResource"); }
	const CGHeroInstance * getHero(ObjectInstanceID) const override { outsideKnowledge("getHero"); }
	const CGTownInstance * getTown(ObjectInstanceID) const override { outsideKnowledge("getTown"); }

	/// Only objects on the player's known tiles exist for the planner; any
	/// other identifier, hidden or not, has no object, and nothing is logged.
	const CGObjectInstance * getObj(ObjectInstanceID id, bool) const override { return getObjInstance(id); }

	const CGObjectInstance * getObjInstance(ObjectInstanceID id) const override
	{
		const auto found = objects.find(id.getNum());
		return found == objects.end() ? nullptr : found->second;
	}

	const CArtifactInstance * getArtInstance(ArtifactInstanceID) const override { outsideKnowledge("getArtInstance"); }

	/// Never null inside the map: a null tile would make the movement cost fall
	/// back to the hero's omniscient callback.
	const TerrainTile * getTile(int3 position, bool) const override { return isInTheMap(position) ? &tiles[position] : nullptr; }
	const TerrainTile * getTileUnchecked(int3 position) const override { return getTile(position, false); }
	const CGObjectInstance * getTopObj(int3) const override { outsideKnowledge("getTopObj"); }
	EDiggingStatus getTileDigStatus(int3, bool) const override { outsideKnowledge("getTileDigStatus"); }

	void calculatePaths(const std::shared_ptr<PathfinderConfig> & config) const override
	{
		CPathfinder(*this, config).calculatePaths();
	}

	int3 guardingCreaturePosition(int3 position) const override { return isInTheMap(position) ? guards[position] : int3(-1, -1, -1); }

	bool checkForVisitableDir(const int3 & source, const int3 & destination) const override
	{
		return grid.visitableFrom(gridPosition(source), gridPosition(destination));
	}

	std::vector<const CGObjectInstance *> getGuardingCreatures(int3 position) const override
	{
		std::vector<const CGObjectInstance *> result;
		if(!isInTheMap(position))
			return result;
		for(const auto id : grid.guardingCreatures(gridPosition(position)))
			result.push_back(getObjInstance(ObjectInstanceID(static_cast<si32>(id))));
		return result;
	}

	bool isTileGuardedUnchecked(int3) const override { outsideKnowledge("isTileGuardedUnchecked"); }

	/// Only a patrolling hero's helper asks, for tile coordinates around its
	/// patrol centre; the planner then clears the patrol anyway.
	void getTilesInRange(FowTilesType & result, const int3 & position, int radius, ETileVisibility mode, std::optional<PlayerColor> other,
		int3::EDistanceFormula formula) const override
	{
		if(other)
			outsideKnowledge("getTilesInRange for a player");
		player.getTilesInRange(result, position, radius, mode, other, formula);
	}

	void getAllTiles(FowTilesType &, std::optional<PlayerColor>, int, const std::function<bool(const TerrainTile *)> &) const override { outsideKnowledge("getAllTiles"); }
	std::vector<ObjectInstanceID> getVisibleTeleportObjects(std::vector<ObjectInstanceID>, PlayerColor) const override { outsideKnowledge("getVisibleTeleportObjects"); }
	std::vector<ObjectInstanceID> getTeleportChannelEntrances(TeleportChannelID, PlayerColor) const override { outsideKnowledge("getTeleportChannelEntrances"); }
	// Teleports are outside the travel modes: no channel has exits or passes.
	std::vector<ObjectInstanceID> getTeleportChannelExits(TeleportChannelID, PlayerColor) const override { return {}; }
	bool isTeleportChannelImpassable(TeleportChannelID, PlayerColor) const override { outsideKnowledge("isTeleportChannelImpassable"); }
	bool isTeleportChannelBidirectional(TeleportChannelID, PlayerColor) const override { return false; }
	bool isTeleportChannelUnidirectional(TeleportChannelID, PlayerColor) const override { return false; }
	bool isTeleportEntrancePassable(const CGTeleport *, PlayerColor) const override { return false; }
#if SCRIPTING_ENABLED
	scripting::Pool * getGlobalContextPool() const override { outsideKnowledge("getGlobalContextPool"); }
#endif

private:
	const CCallback & player;
	PlayerColor color;
	int3 size;
	KnowledgeGrid grid;
	MapTilesStorage<TerrainTile> tiles; ///< known tiles with only visible objects listed, otherwise rock
	MapTilesStorage<int3> guards;
	TerrainTile unknown;
	std::unordered_map<si32, const CGObjectInstance *> objects; ///< every visible object on a known tile

	void capture(const int3 & position)
	{
		if(!player.isVisible(position))
		{
			tiles[position] = unknown;
			return;
		}
		const auto * real = player.getTile(position, false);
		if(!real)
			throw std::runtime_error("A known tile has no terrain");
		TerrainTile tile = *real;
		tile.blockingObjects.clear();
		tile.visitableObjects.clear();
		auto & known = grid.at(gridPosition(position));
		known.known = true;
		known.water = real->isWater();
		known.entrable = real->entrableTerrain();
		// Every list keeps its order: the top visitable object is the last one.
		for(const auto id : real->blockingObjects)
			if(visible(id))
				tile.blockingObjects.push_back(id);
		for(const auto id : real->visitableObjects)
		{
			const auto * object = visible(id);
			if(!object)
				continue;
			tile.visitableObjects.push_back(id);
			known.visitable.push_back({id.getNum(), object->ID == Obj::MONSTER, vstd::contains(tile.blockingObjects, id),
				object->isBlockedVisitable(), visitDirections(*object)});
		}
		tiles[position] = std::move(tile);
	}

	/// An object the player sees, never an event, recorded for getObjInstance.
	const CGObjectInstance * visible(ObjectInstanceID id)
	{
		const auto found = objects.find(id.getNum());
		if(found != objects.end())
			return found->second;
		const auto * object = playerVisibleObject(player, id);
		if(object)
			objects.emplace(id.getNum(), object);
		return object;
	}
};

/// Land accessibility uses only visible objects. The native air/water rules
/// consume those same filtered tiles and this player's fog map; neither can
/// route through unknown tiles. Sailing nodes remain absent.
class KnowledgeNodeStorage final : public NodeStorage
{
public:
	KnowledgeNodeStorage(CPathsInfo & paths, const CGHeroInstance * hero, const KnowledgeMap & knowledge)
		: NodeStorage(paths, hero), knowledge(knowledge), hero(hero)
	{
	}

	void initialize(const PathfinderOptions & options, const IGameInfoCallback &) override
	{
		const auto size = knowledge.getMapSize();
		const auto owner = hero->getOwner();
		const auto & fog = knowledge.getPlayerTeam(owner)->fogOfWarMap;
		for(int z = 0; z < size.z; ++z)
			for(int x = 0; x < size.x; ++x)
				for(int y = 0; y < size.y; ++y)
				{
					const int3 position(x, y, z);
					const auto & tile = *knowledge.getTile(position, false);
					if(tile.isLand())
						getNode(position, EPathfindingLayer::LAND)->update(position, EPathfindingLayer::LAND, knowledge.landAccessibility(position));
					if(options.useFlying && (tile.isLand() || tile.isWater()))
						getNode(position, EPathfindingLayer::AIR)->update(position, EPathfindingLayer::AIR,
							PathfinderUtil::evaluateAccessibility<EPathfindingLayer::AIR>(position, tile, fog, owner, knowledge));
					if(options.useWaterWalking && tile.isWater())
						getNode(position, EPathfindingLayer::WATER)->update(position, EPathfindingLayer::WATER,
							PathfinderUtil::evaluateAccessibility<EPathfindingLayer::WATER>(position, tile, fog, owner, knowledge));
				}
	}

	std::vector<CGPathNode *> calculateTeleportations(const PathNodeInfo &, const PathfinderConfig *, const CPathfinderHelper *) override
	{
		return {};
	}

private:
	const KnowledgeMap & knowledge;
	const CGHeroInstance * hero;
};

/// Refuses sailing before any rule can commit it. Embark options alone do not
/// disable the sailing layer; active flight/water walking remain native layers.
class SupportedLayerRule final : public IPathfindingRule
{
public:
	void process(const PathNodeInfo & source, CDestinationNodeInfo & destination, const PathfinderConfig *, CPathfinderHelper *) const override
	{
		if(source.node->layer == EPathfindingLayer::SAIL || destination.node->layer == EPathfindingLayer::SAIL)
			destination.blocked = true;
	}
};

/// DestinationActionRule for land moves, with town and garrison passability
/// taken from ownership instead of their hidden garrisons.
class KnowledgeDestinationActionRule final : public IPathfindingRule
{
public:
	void process(const PathNodeInfo &, CDestinationNodeInfo & destination, const PathfinderConfig *, CPathfinderHelper * helper) const override
	{
		if(destination.action != EPathNodeAction::UNKNOWN)
			return;
		// Match DestinationActionRule: special layers transit without visits or
		// battles, even above a visible guard. Landing follows the land rules.
		if(destination.node->layer != EPathfindingLayer::LAND)
		{
			destination.action = EPathNodeAction::NORMAL;
			return;
		}
		auto action = EPathNodeAction::NORMAL;
		if(destination.isNodeObjectVisitable())
		{
			const auto * object = destination.nodeObject;
			const bool enemy = destination.objectRelations == PlayerRelations::ENEMIES;
			if(object->ID == Obj::BOAT)
			{
				destination.blocked = true; // embarking is outside the travel modes
				return;
			}
			if(destination.nodeHero)
				action = destination.heroRelations == PlayerRelations::ENEMIES ? EPathNodeAction::BATTLE : EPathNodeAction::BLOCKING_VISIT;
			else if(object->ID == Obj::TOWN)
				action = enemy ? EPathNodeAction::BATTLE : EPathNodeAction::VISIT;
			else if(object->ID == Obj::GARRISON || object->ID == Obj::GARRISON2)
			{
				if(enemy)
					action = EPathNodeAction::BATTLE;
				else if(destination.guarded)
					action = EPathNodeAction::BATTLE;
			}
			else if(object->ID == Obj::BORDER_GATE)
			{
				if(!object->passableFor(helper->owner))
					action = EPathNodeAction::BLOCKING_VISIT;
				else if(destination.guarded)
					action = EPathNodeAction::BATTLE;
			}
			else if(destination.isGuardianTile)
				action = EPathNodeAction::BATTLE;
			else if(object->isBlockedVisitable())
				action = EPathNodeAction::BLOCKING_VISIT;
			if(action == EPathNodeAction::NORMAL)
				action = destination.guarded ? EPathNodeAction::BATTLE : EPathNodeAction::VISIT;
		}
		else if(destination.guarded)
			action = EPathNodeAction::BATTLE;
		destination.action = action;
	}
};

/// One hero's native movement configuration over a knowledge map.
class KnowledgeConfig final : public PathfinderConfig
{
public:
	KnowledgeConfig(std::shared_ptr<INodeStorage> storage, const KnowledgeMap & knowledge, const CGHeroInstance * hero)
		: PathfinderConfig(std::move(storage), knowledge, {
			std::make_shared<SupportedLayerRule>(),
			std::make_shared<LayerTransitionRule>(),
			std::make_shared<KnowledgeDestinationActionRule>(),
			std::make_shared<MovementToDestinationRule>(),
			std::make_shared<MovementCostRule>(),
			std::make_shared<MovementAfterDestinationRule>()})
		, hero(hero)
	{
		const auto turn = hero->getTurnInfo(0);
		options.useFlying = turn->hasFlyingMovement();
		options.useWaterWalking = turn->hasWaterWalking();
		options.ignoreGuards = false;
		options.useEmbarkAndDisembark = false;
		options.useTeleportTwoWay = false;
		options.useTeleportOneWay = false;
		options.useTeleportOneWayRandom = false;
		options.useTeleportWhirlpool = false;
		options.forceUseTeleportWhirlpool = false;
		options.useCastleGate = false;
		options.lightweightFlyingMode = false;
		options.oneTurnSpecialLayersLimit = true; // the header warns that false crashes
		options.turnLimit = std::numeric_limits<uint8_t>::max(); // a far target still has a current-turn prefix
		options.canUseCast = false;
		options.allowLayerTransitioningAfterBattle = false;
	}

	CPathfinderHelper * getOrCreatePathfinderHelper(const PathNodeInfo &, const IGameInfoCallback & gameInfo) override
	{
		if(!helper)
		{
			helper = std::make_unique<CPathfinderHelper>(gameInfo, hero, options);
			// The server never enforces a map patrol on a player's hero.
			helper->patrolState = CPathfinderHelper::PATROL_NONE;
			helper->patrolTiles.clear();
		}
		return helper.get();
	}

private:
	const CGHeroInstance * hero;
	std::unique_ptr<CPathfinderHelper> helper;
};
}

RoutePlan planRoute(const CCallback & player, const CGHeroInstance & hero, const int3 & destination)
{
	if(!player.getPlayerID() || *player.getPlayerID() != hero.getOwner())
		throw std::logic_error("Knowledge routing requires the owned hero's player callback");
	RoutePlan plan;
	if(hero.inBoat())
		return plan; // sailing is outside the travel modes
	const KnowledgeMap knowledge(player, hero.getOwner());
	if(!knowledge.isInTheMap(destination) || destination == hero.visitablePos())
		throw std::logic_error("Plan only a different destination inside the map");
	// Fresh paths, storage and configuration per plan: reused nodes keep stale costs.
	CPathsInfo paths(knowledge.getMapSize(), &hero);
	auto storage = std::make_shared<KnowledgeNodeStorage>(paths, &hero, knowledge);
	auto config = std::make_shared<KnowledgeConfig>(storage, knowledge, &hero);
	CPathfinder(knowledge, config).calculatePaths();
	CGPath path;
	if(!paths.getPath(path, destination))
		return plan;
	for(size_t i = 0; i < path.nodes.size(); ++i)
	{
		const auto & node = path.nodes[i];
		const bool origin = i + 1 == path.nodes.size();
		const bool supported = node.action == EPathNodeAction::NORMAL || node.action == EPathNodeAction::VISIT
			|| node.action == EPathNodeAction::BLOCKING_VISIT || node.action == EPathNodeAction::BATTLE;
		const bool supportedLayer = node.layer == EPathfindingLayer::LAND || node.layer == EPathfindingLayer::AIR
			|| node.layer == EPathfindingLayer::WATER;
		if(!supportedLayer || !(origin || supported))
			throw std::logic_error("The knowledge planner produced an unsupported movement step");
	}
	plan.known = true;
	size_t landedPrefix = 0;
	for(size_t i = path.nodes.size() - 1; i-- > 0;)
	{
		const auto & node = path.nodes[i];
		if(node.turns != 0)
			break;
		const bool transit = node.layer == EPathfindingLayer::AIR || node.layer == EPathfindingLayer::WATER;
		plan.steps.push_back({node.coord, transit});
		if(!transit)
			landedPrefix = plan.steps.size();
	}
	// A move choice must finish where the hero can stand. The native one-turn
	// special-layer rule already enforces this for ordinary routes; retaining
	// only a landed prefix also prevents a turn cutoff from stranding the hero.
	plan.steps.resize(landedPrefix);
	plan.complete = plan.steps.size() + 1 == path.nodes.size();
	return plan;
}
}
