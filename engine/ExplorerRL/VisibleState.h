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

struct VisibleSnapshot
{
	int width = 0;
	int height = 0;
	int levels = 0;
	std::vector<uint8_t> tiles; ///< 1 where the player's team sees the tile, x + width * (y + height * z)
	std::map<int64_t, VisibleObject> objects; ///< only objects on visible tiles that are visible themselves
};

/// One ordered difference between consecutive captures. A disappearance only
/// states that the object is no longer visible: it has no cause, no later
/// position and no other fact, so a hidden move or death is never disclosed.
struct VisibleChange
{
	std::vector<VisibleObject> appeared;
	std::vector<VisibleObject> changed;
	std::vector<int64_t> disappeared;
	std::vector<std::array<int, 3>> revealed;
	std::vector<std::array<int, 3>> hidden;

	bool empty() const
	{
		return appeared.empty() && changed.empty() && disappeared.empty() && revealed.empty() && hidden.empty();
	}
};

/// World requirements: width(), height(), levels(), tileVisible(x, y, z) and
/// visibleObjectsAt(x, y, z, std::vector<VisibleObject> &). The latter is called
/// only for tiles the player sees and must itself skip objects it cannot see.
/// Hidden tiles are never queried for objects.
template<typename World>
VisibleSnapshot captureVisible(const World & world)
{
	VisibleSnapshot snapshot;
	snapshot.width = world.width();
	snapshot.height = world.height();
	snapshot.levels = world.levels();
	snapshot.tiles.assign(static_cast<size_t>(snapshot.width) * snapshot.height * snapshot.levels, 0);
	std::vector<VisibleObject> objects;
	for(int z = 0; z < snapshot.levels; ++z)
		for(int y = 0; y < snapshot.height; ++y)
			for(int x = 0; x < snapshot.width; ++x)
			{
				if(!world.tileVisible(x, y, z))
					continue;
				snapshot.tiles[x + snapshot.width * (y + snapshot.height * z)] = 1;
				objects.clear();
				world.visibleObjectsAt(x, y, z, objects);
				for(const auto & object : objects)
					snapshot.objects.emplace(object.id, object); // a multi-tile object keeps its first visible entry
			}
	return snapshot;
}

/// Keeps the previous capture of one player and reports the next difference.
/// The first capture is reported against an empty view, so the ordered change
/// stream alone reconstructs the visible objects and tiles at every capture.
class VisibleTracker
{
	std::optional<VisibleSnapshot> previous;

public:
	VisibleChange update(VisibleSnapshot current)
	{
		VisibleChange change;
		const VisibleSnapshot empty{current.width, current.height, current.levels,
			std::vector<uint8_t>(current.tiles.size(), 0), {}};
		const VisibleSnapshot & before = previous ? *previous : empty;
		if(before.tiles.size() != current.tiles.size() || before.width != current.width || before.height != current.height)
			throw std::runtime_error("Visible map dimensions changed during an episode");
		for(size_t index = 0; index < current.tiles.size(); ++index)
		{
			if(before.tiles[index] == current.tiles[index])
				continue;
			const int x = static_cast<int>(index % current.width);
			const int y = static_cast<int>((index / current.width) % current.height);
			const int z = static_cast<int>(index / (static_cast<size_t>(current.width) * current.height));
			(current.tiles[index] ? change.revealed : change.hidden).push_back({x, y, z});
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
		previous = std::move(current);
		return change;
	}
};
}
