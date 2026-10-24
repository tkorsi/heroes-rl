#pragma once

#include "VisibleState.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

/// The map-context facts of adventure-windows-v1 and their ordered changes.
/// Unlike visible_update-v2 it adds each visible tile's road and lists every
/// owned hero and town with its engine ID for routing, and it never carries a
/// hero's army or the mine count: armies belong to windows, which are read
/// only while open. This header has no engine dependency: ExplorerRL supplies
/// a world backed by the player's own callback, and tests supply synthetic ones.
namespace explorer_rl
{
struct ContextTile
{
	int terrain = 0;
	int road = 0;
	bool blocked = false;
	bool visitable = false;

	bool operator==(const ContextTile &) const = default;
};

struct ContextTileFact
{
	std::array<int, 3> position{};
	ContextTile tile;

	bool operator==(const ContextTileFact &) const = default;
};

struct OwnHero
{
	int64_t id = -1; ///< routing only; the adapter replaces it with an opaque handle
	std::array<int, 3> position{};
	int movement = 0;
	int movementMax = 0;

	bool operator==(const OwnHero &) const = default;
};

struct OwnTown
{
	int64_t id = -1; ///< routing only
	std::array<int, 3> position{};

	bool operator==(const OwnTown &) const = default;
};

/// What the map context shows of the player's own kingdom.
struct ContextOwn
{
	std::vector<int64_t> resources;
	std::vector<OwnHero> heroes; ///< ascending ID
	std::vector<OwnTown> towns; ///< ascending ID

	bool operator==(const ContextOwn &) const = default;
};

struct ContextSnapshot
{
	int width = 0;
	int height = 0;
	int levels = 0;
	std::vector<uint8_t> tiles; ///< 1 where the player's team sees the tile, x + width * (y + height * z)
	std::vector<ContextTile> facts; ///< meaningful only where tiles is 1
	std::map<int64_t, VisibleObject> objects; ///< only visible objects on visible tiles
	ContextOwn own;
};

/// One ordered difference between consecutive captures, with the same
/// disclosure rules as VisibleChange: a disappearance has no cause, and a
/// hidden tile carries no facts. An owned hero that is no longer owned is
/// reported once, by ID, as lost.
struct ContextChange
{
	std::vector<VisibleObject> appeared;
	std::vector<VisibleObject> changed;
	std::vector<int64_t> disappeared;
	std::vector<ContextTileFact> revealed;
	std::vector<ContextTileFact> tilesChanged;
	std::vector<std::array<int, 3>> hidden;
	std::optional<ContextOwn> own; ///< the complete own state, present when it changed
	std::vector<int64_t> heroesLost;

	bool empty() const
	{
		return appeared.empty() && changed.empty() && disappeared.empty() && revealed.empty()
			&& tilesChanged.empty() && hidden.empty() && !own && heroesLost.empty();
	}
};

/// World requirements: width(), height(), levels(), tileVisible(x, y, z),
/// tileFacts(x, y, z) returning a ContextTile, visibleObjectsAt(x, y, z,
/// std::vector<VisibleObject> &) and ownState() returning a ContextOwn.
/// Facts and objects are requested only for tiles the player sees.
template<typename World>
ContextSnapshot captureContext(const World & world)
{
	ContextSnapshot snapshot;
	snapshot.width = world.width();
	snapshot.height = world.height();
	snapshot.levels = world.levels();
	const size_t area = static_cast<size_t>(snapshot.width) * snapshot.height * snapshot.levels;
	snapshot.tiles.assign(area, 0);
	snapshot.facts.assign(area, ContextTile{});
	std::vector<VisibleObject> objects;
	for(int z = 0; z < snapshot.levels; ++z)
		for(int y = 0; y < snapshot.height; ++y)
			for(int x = 0; x < snapshot.width; ++x)
			{
				if(!world.tileVisible(x, y, z))
					continue;
				const size_t index = x + snapshot.width * (y + static_cast<size_t>(snapshot.height) * z);
				snapshot.tiles[index] = 1;
				snapshot.facts[index] = world.tileFacts(x, y, z);
				objects.clear();
				world.visibleObjectsAt(x, y, z, objects);
				for(const auto & object : objects)
					snapshot.objects.emplace(object.id, object); // a multi-tile object keeps its first visible entry
			}
	snapshot.own = world.ownState();
	const auto byID = [](const auto & a, const auto & b) { return a.id < b.id; };
	std::sort(snapshot.own.heroes.begin(), snapshot.own.heroes.end(), byID);
	std::sort(snapshot.own.towns.begin(), snapshot.own.towns.end(), byID);
	return snapshot;
}

/// Digest of one capture for engine-side evidence only (it contains engine
/// IDs), in the canonical encoding of snapshotDigest with roads and the
/// contextual own state.
inline uint64_t contextDigest(const ContextSnapshot & snapshot)
{
	Digest digest;
	digest.add(snapshot.width);
	digest.add(snapshot.height);
	digest.add(snapshot.levels);
	for(size_t index = 0; index < snapshot.tiles.size(); ++index)
	{
		if(!snapshot.tiles[index])
			continue;
		const auto & tile = snapshot.facts[index];
		for(const int64_t value : {static_cast<int64_t>(index), int64_t{tile.terrain}, int64_t{tile.road},
				int64_t{tile.blocked}, int64_t{tile.visitable}})
			digest.add(value);
	}
	digest.add(static_cast<int64_t>(snapshot.objects.size()));
	for(const auto & [id, object] : snapshot.objects)
		for(const int64_t value : {id, int64_t{object.x}, int64_t{object.y}, int64_t{object.z},
				int64_t{object.type}, int64_t{object.subtype}, int64_t{object.owner}})
			digest.add(value);
	const auto & own = snapshot.own;
	digest.add(static_cast<int64_t>(own.resources.size()));
	for(const auto amount : own.resources)
		digest.add(amount);
	digest.add(static_cast<int64_t>(own.heroes.size()));
	for(const auto & hero : own.heroes)
		for(const int64_t value : {hero.id, int64_t{hero.position[0]}, int64_t{hero.position[1]}, int64_t{hero.position[2]},
				int64_t{hero.movement}, int64_t{hero.movementMax}})
			digest.add(value);
	digest.add(static_cast<int64_t>(own.towns.size()));
	for(const auto & town : own.towns)
		for(const int64_t value : {town.id, int64_t{town.position[0]}, int64_t{town.position[1]}, int64_t{town.position[2]}})
			digest.add(value);
	return digest.value();
}

/// Keeps one player's previous capture and reports the next difference. The
/// first capture is reported against an empty view, so the ordered change
/// stream alone reconstructs the map context at every capture.
class ContextTracker
{
	std::optional<ContextSnapshot> previous;

public:
	const std::optional<ContextSnapshot> & latest() const { return previous; }

	ContextChange update(ContextSnapshot current)
	{
		ContextChange change;
		const size_t area = current.tiles.size();
		if(current.facts.size() != area)
			throw std::runtime_error("Context tile facts do not cover the map");
		const ContextSnapshot empty{current.width, current.height, current.levels,
			std::vector<uint8_t>(area, 0), std::vector<ContextTile>(area), {}, {}};
		const ContextSnapshot & before = previous ? *previous : empty;
		if(before.tiles.size() != area || before.width != current.width || before.height != current.height)
			throw std::runtime_error("Context map dimensions changed during an episode");
		for(size_t index = 0; index < area; ++index)
		{
			if(!before.tiles[index] && !current.tiles[index])
				continue;
			const int x = static_cast<int>(index % current.width);
			const int y = static_cast<int>((index / current.width) % current.height);
			const int z = static_cast<int>(index / (static_cast<size_t>(current.width) * current.height));
			if(!current.tiles[index])
				change.hidden.push_back({x, y, z});
			else if(!before.tiles[index])
				change.revealed.push_back({{x, y, z}, current.facts[index]});
			else if(!(before.facts[index] == current.facts[index]))
				change.tilesChanged.push_back({{x, y, z}, current.facts[index]});
		}
		for(const auto & [id, object] : current.objects)
		{
			const auto old = before.objects.find(id);
			if(old == before.objects.end())
				change.appeared.push_back(object);
			else if(!(old->second == object))
				change.changed.push_back(object);
		}
		for(const auto & [id, object] : before.objects)
			if(!current.objects.contains(id))
				change.disappeared.push_back(id);
		if(!previous || !(previous->own == current.own))
			change.own = current.own;
		for(const auto & hero : before.own.heroes)
			if(std::none_of(current.own.heroes.begin(), current.own.heroes.end(), [&hero](const auto & now) { return now.id == hero.id; }))
				change.heroesLost.push_back(hero.id);
		previous = std::move(current);
		return change;
	}
};
}
