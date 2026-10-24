#pragma once

#include "../../lib/callback/CCallback.h"
#include "../../lib/mapObjects/CGHeroInstance.h"
#include "../../lib/mapObjects/CGTownInstance.h"
#include "../../lib/mapObjects/army/CStackInstance.h"
#include "../../lib/mapping/TerrainTile.h"
#include "ContextualState.h"
#include "VisibleState.h"

/// The player-visible world as one player's callback reports it. ExplorerRL
/// captures through it, and the replay verifier recomputes the same captures
/// from the replayed authoritative state, so both use exactly this code.
namespace explorer_rl
{
/// An object the player can see, or null. Event objects are hidden from
/// players even on visible tiles, as in CGameInfoCallback::getVisitableObjs.
inline const CGObjectInstance * playerVisibleObject(const CCallback & callback, ObjectInstanceID objectID)
{
	const auto * object = callback.getObj(objectID, false);
	return object && object->ID != Obj::EVENT && callback.isVisible(object) ? object : nullptr;
}

/// Terrain and flags of a tile the player sees, from the objects it can see there.
inline VisibleTile visibleTileFacts(const CCallback & callback, const TerrainTile & tile)
{
	auto anyVisible = [&callback](const std::vector<ObjectInstanceID> & objects)
	{
		return std::any_of(objects.begin(), objects.end(), [&callback](const auto objectID)
		{
			return playerVisibleObject(callback, objectID) != nullptr;
		});
	};
	return {tile.getTerrainID().getNum(), anyVisible(tile.blockingObjects), anyVisible(tile.visitableObjects)};
}

/// The player's own callback as a VisibleState world. Objects are listed only
/// for tiles its team sees, and only when the callback reports them visible.
struct CallbackWorld
{
	const CCallback & callback;
	PlayerColor player;
	int3 size;

	int width() const { return size.x; }
	int height() const { return size.y; }
	int levels() const { return size.z; }
	bool tileVisible(int x, int y, int z) const { return callback.isVisible(int3(x, y, z)); }

	VisibleTile tileFacts(int x, int y, int z) const
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile)
			throw std::runtime_error("A visible tile has no terrain");
		return visibleTileFacts(callback, *tile);
	}

	void visibleObjectsAt(int x, int y, int z, std::vector<VisibleObject> & out) const
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile)
			return;
		for(const auto objectID : tile->visitableObjects)
			if(const auto * object = playerVisibleObject(callback, objectID))
				out.push_back({object->id.getNum(), x, y, z, object->ID.getNum(), object->subID.getNum(), object->getOwner().getNum()});
	}

	OwnState ownState() const
	{
		OwnState own;
		for(int resource = 0; resource < GameConstants::RESOURCE_QUANTITY; ++resource)
			own.resources.push_back(callback.getResourceAmount(GameResID(resource)));
		for(const auto * hero : callback.getHeroesInfo())
		{
			if(hero->getOwner() != player)
				continue;
			const auto position = hero->visitablePos();
			own.hero = std::array<int, 3>{position.x, position.y, position.z};
			own.movement = hero->movementPointsRemaining();
			own.movementMax = hero->movementPointsLimit();
			for(const auto & [slot, stack] : hero->Slots())
				own.army.push_back({slot.getNum(), stack->getCreatureID().getNum(), stack->getCount()});
			break;
		}
		for(const auto * town : callback.getTownsInfo())
			if(town->getOwner() == player)
			{
				const auto position = town->visitablePos();
				own.towns.push_back({position.x, position.y, position.z});
			}
		std::sort(own.towns.begin(), own.towns.end());
		for(const auto * object : callback.getMyObjects())
			if(object->getOwner() == player && object->ID == Obj::MINE)
				++own.mines;
		return own;
	}
};

/// The player's own callback as an adventure-windows-v1 map context: visible
/// tiles with their roads, the same visible objects, and every owned hero and
/// town without armies.
struct CallbackContextWorld
{
	const CCallback & callback;
	PlayerColor player;
	int3 size;

	int width() const { return size.x; }
	int height() const { return size.y; }
	int levels() const { return size.z; }
	bool tileVisible(int x, int y, int z) const { return callback.isVisible(int3(x, y, z)); }

	ContextTile tileFacts(int x, int y, int z) const
	{
		const auto * tile = callback.getTile(int3(x, y, z), false);
		if(!tile)
			throw std::runtime_error("A visible tile has no terrain");
		const auto facts = visibleTileFacts(callback, *tile);
		return {facts.terrain, tile->getRoadID().getNum(), facts.blocked, facts.visitable};
	}

	void visibleObjectsAt(int x, int y, int z, std::vector<VisibleObject> & out) const
	{
		CallbackWorld{callback, player, size}.visibleObjectsAt(x, y, z, out);
	}

	ContextOwn ownState() const
	{
		ContextOwn own;
		for(int resource = 0; resource < GameConstants::RESOURCE_QUANTITY; ++resource)
			own.resources.push_back(callback.getResourceAmount(GameResID(resource)));
		for(const auto * hero : callback.getHeroesInfo())
			if(hero->getOwner() == player)
			{
				const auto position = hero->visitablePos();
				own.heroes.push_back({hero->id.getNum(), {position.x, position.y, position.z},
					hero->movementPointsRemaining(), hero->movementPointsLimit()});
			}
		for(const auto * town : callback.getTownsInfo())
			if(town->getOwner() == player)
			{
				const auto position = town->visitablePos();
				own.towns.push_back({town->id.getNum(), {position.x, position.y, position.z}});
			}
		return own;
	}
};
}
