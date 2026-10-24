#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

/// Player-visible map facts and their ordered changes between policy decisions.
/// This header has no engine dependency: ExplorerRL supplies a World adapter
/// backed by the player's own callback, and tests supply synthetic worlds.
namespace explorer_rl
{
/// Facts about one visitable object as the player can see it at one moment.
/// The engine object ID is routing metadata for the Python adapter, which
/// replaces it with an opaque per-player handle before any learner sees it.
struct VisibleObject
{
	int64_t id = -1;
	int x = 0;
	int y = 0;
	int z = 0;
	int type = 0;
	int subtype = 0;
	int owner = 0;

	bool operator==(const VisibleObject &) const = default;
};

/// Terrain and flags of a tile the player sees. The flags are derived only
/// from objects the player can see on that tile.
struct VisibleTile
{
	int terrain = 0;
	bool blocked = false;
	bool visitable = false;

	bool operator==(const VisibleTile &) const = default;
};

struct TileFact
{
	std::array<int, 3> position{};
	VisibleTile tile;

	bool operator==(const TileFact &) const = default;
};

struct ArmyStack
{
	int slot = 0;
	int creature = 0;
	int count = 0;

	bool operator==(const ArmyStack &) const = default;
};

/// What the player knows about its own kingdom under normal rules, independent
/// of fog: its hero, resources, hero army, towns and number of owned mines.
struct OwnState
{
	std::optional<std::array<int, 3>> hero;
	int movement = 0;
	int movementMax = 0;
	std::vector<int64_t> resources;
	std::vector<ArmyStack> army; ///< ascending slot order
	std::vector<std::array<int, 3>> towns; ///< ascending position order
	int mines = 0;

	bool operator==(const OwnState &) const = default;
};

struct VisibleSnapshot
{
	int width = 0;
	int height = 0;
	int levels = 0;
	std::vector<uint8_t> tiles; ///< 1 where the player's team sees the tile, x + width * (y + height * z)
	std::vector<VisibleTile> facts; ///< meaningful only where tiles is 1
	std::map<int64_t, VisibleObject> objects; ///< only objects on visible tiles that are visible themselves
	OwnState own;
};

/// One ordered difference between consecutive captures. A disappearance only
/// states that the object is no longer visible: it has no cause, no later
/// position and no other fact, so a hidden move or death is never disclosed.
/// A hidden tile likewise carries no facts, and a tile's facts are reported
/// again only while the player sees it.
struct VisibleChange
{
	std::vector<VisibleObject> appeared;
	std::vector<VisibleObject> changed;
	std::vector<int64_t> disappeared;
	std::vector<TileFact> revealed;
	std::vector<TileFact> tilesChanged; ///< still visible tiles whose facts changed
	std::vector<std::array<int, 3>> hidden;
	std::optional<OwnState> own; ///< the complete own state, present when it changed

	bool empty() const
	{
		return appeared.empty() && changed.empty() && disappeared.empty() && revealed.empty()
			&& tilesChanged.empty() && hidden.empty() && !own;
	}
};

/// World requirements: width(), height(), levels(), tileVisible(x, y, z),
/// tileFacts(x, y, z), visibleObjectsAt(x, y, z, std::vector<VisibleObject> &)
/// and ownState(). Tile facts and objects are requested only for tiles the
/// player sees, and visibleObjectsAt must itself skip objects it cannot see.
template<typename World>
VisibleSnapshot captureVisible(const World & world)
{
	VisibleSnapshot snapshot;
	snapshot.width = world.width();
	snapshot.height = world.height();
	snapshot.levels = world.levels();
	const size_t area = static_cast<size_t>(snapshot.width) * snapshot.height * snapshot.levels;
	snapshot.tiles.assign(area, 0);
	snapshot.facts.assign(area, VisibleTile{});
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
	return snapshot;
}

/// Keeps the previous capture of one player and reports the next difference.
/// The first capture is reported against an empty view, so the ordered change
/// stream alone reconstructs the visible objects, tiles and own state at every
/// capture.
class VisibleTracker
{
	std::optional<VisibleSnapshot> previous;

public:
	VisibleChange update(VisibleSnapshot current)
	{
		VisibleChange change;
		const size_t area = current.tiles.size();
		if(current.facts.size() != area)
			throw std::runtime_error("Visible tile facts do not cover the map");
		const VisibleSnapshot empty{current.width, current.height, current.levels,
			std::vector<uint8_t>(area, 0), std::vector<VisibleTile>(area), {}, {}};
		const VisibleSnapshot & before = previous ? *previous : empty;
		if(before.tiles.size() != area || before.width != current.width || before.height != current.height)
			throw std::runtime_error("Visible map dimensions changed during an episode");
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
		previous = std::move(current);
		return change;
	}
};
}
