#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

/// One player's map knowledge as the contextual route planner reads it, and
/// the zone-of-control and visit-direction rules VCMI 1.7.5 applies, evaluated
/// over that knowledge only. This header has no engine dependency:
/// KnowledgePlanner fills the grid from the player's own callback, and tests
/// fill it from synthetic worlds.
namespace explorer_rl
{
using GridPosition = std::array<int, 3>;

/// One object a player can see on one tile.
struct KnownObject
{
	int64_t id = -1;
	bool monster = false; ///< a wandering monster, which guards its neighbourhood
	bool blockingHere = false; ///< the tile's blocking list holds it too
	bool blockedVisitable = false; ///< visited from a neighbouring tile, never entered
	/// ObjectTemplate::visitDir: the neighbouring directions it can be visited
	/// from, bit 1 north-west, 2 north, 4 north-east, 8 east, 16 south-east,
	/// 32 south, 64 south-west and 128 west.
	uint8_t visitDir = 0;
};

/// One tile of a player's knowledge. An unknown tile keeps the defaults: it is
/// never entrable and holds no objects, like the impassable rock the planner
/// substitutes for it, so nothing behind the fog can guard or block a route.
struct KnownTile
{
	bool known = false;
	bool water = false;
	bool entrable = false; ///< passable terrain, which rock is not
	std::vector<KnownObject> visitable; ///< visible visitable objects, bottom to top
};

class KnowledgeGrid
{
public:
	KnowledgeGrid(int width, int height, int levels)
		: width(width), height(height), levels(levels)
	{
		if(width <= 0 || height <= 0 || levels <= 0)
			throw std::invalid_argument("A knowledge grid needs positive dimensions");
		tiles.resize(static_cast<size_t>(width) * height * levels);
	}

	bool inMap(const GridPosition & position) const
	{
		return position[0] >= 0 && position[0] < width && position[1] >= 0 && position[1] < height
			&& position[2] >= 0 && position[2] < levels;
	}

	KnownTile & at(const GridPosition & position) { return tiles.at(index(position)); }
	const KnownTile & at(const GridPosition & position) const { return tiles.at(index(position)); }

	/// CMap::checkForVisitableDir(src, tile(dst), dst): whether a hero standing
	/// on src can step onto or attack dst, given dst's blocking visitable objects.
	bool visitableFrom(const GridPosition & source, const GridPosition & destination) const
	{
		const auto & tile = at(destination);
		if(!tile.entrable)
			return false;
		for(const auto & object : tile.visitable)
			if(object.blockingHere && !visitableFromDirection(object.visitDir, source[0] - destination[0], source[1] - destination[1]))
				return false;
		return true;
	}

	/// CMap::guardingCreaturePosition over knowledge: the known monster whose
	/// zone of control covers the position, giving one standing there priority,
	/// then scanning the neighbourhood column by column from the north-west.
	std::optional<GridPosition> guard(const GridPosition & position) const
	{
		if(!inMap(position))
			return std::nullopt;
		const auto & tile = at(position);
		for(const auto & object : tile.visitable)
			if(object.monster)
				return position;
		for(int dx = -1; dx <= 1; ++dx)
			for(int dy = -1; dy <= 1; ++dy)
			{
				const GridPosition neighbour{position[0] + dx, position[1] + dy, position[2]};
				if(!inMap(neighbour))
					continue;
				const auto & candidate = at(neighbour);
				if(candidate.visitable.empty() || candidate.water != tile.water)
					continue;
				for(const auto & object : candidate.visitable)
					if(object.monster && visitableFrom(neighbour, position))
						return neighbour;
			}
		return std::nullopt;
	}

	/// CGameState::guardingCreatures over knowledge, as object IDs. It keeps the
	/// engine's double count of a monster standing on the position itself: the
	/// pathfinder only asks whether more than one guard covers a tile.
	std::vector<int64_t> guardingCreatures(const GridPosition & position) const
	{
		std::vector<int64_t> guards;
		if(!guard(position))
			return guards;
		const auto & tile = at(position);
		for(const auto & object : tile.visitable)
			if(object.blockedVisitable && object.monster)
				guards.push_back(object.id);
		for(int dx = -1; dx <= 1; ++dx)
			for(int dy = -1; dy <= 1; ++dy)
			{
				const GridPosition neighbour{position[0] + dx, position[1] + dy, position[2]};
				if(!inMap(neighbour))
					continue;
				const auto & candidate = at(neighbour);
				if(candidate.visitable.empty() || candidate.water != tile.water)
					continue;
				for(const auto & object : candidate.visitable)
					if(object.monster && visitableFrom(neighbour, position))
						guards.push_back(object.id);
			}
		return guards;
	}

	/// ObjectTemplate::isVisitableFrom for a source offset (dx, dy) from the object.
	static bool visitableFromDirection(uint8_t visitDir, int dx, int dy)
	{
		const int column = dx < 0 ? 0 : dx == 0 ? 1 : 2;
		const int row = dy < 0 ? 0 : dy == 0 ? 1 : 2;
		const int directions[3][3] = {
			{visitDir & 1, visitDir & 2, visitDir & 4},
			{visitDir & 128, 1, visitDir & 8},
			{visitDir & 64, visitDir & 32, visitDir & 16}
		};
		return directions[row][column] != 0;
	}

private:
	int width;
	int height;
	int levels;
	std::vector<KnownTile> tiles; ///< x + width * (y + height * z), as in VisibleSnapshot

	size_t index(const GridPosition & position) const
	{
		if(!inMap(position))
			throw std::out_of_range("Position outside the knowledge grid");
		return static_cast<size_t>(position[0]) + static_cast<size_t>(width) * (position[1] + static_cast<size_t>(height) * position[2]);
	}
};

/// One node of a planned path, as CPathsInfo::getPath lists it: destination
/// first, the hero's origin last.
struct PlannedNode
{
	GridPosition position{};
	int turns = 0; ///< 0 for this turn
};

/// The current-turn prefix of a planned path, origin excluded, in walking
/// order. It stops before the first node that needs another turn.
inline std::vector<GridPosition> currentTurnPrefix(const std::vector<PlannedNode> & path)
{
	std::vector<GridPosition> steps;
	if(path.size() < 2)
		return steps;
	for(size_t i = path.size() - 1; i-- > 0;)
	{
		if(path[i].turns != 0)
			break;
		steps.push_back(path[i].position);
	}
	return steps;
}
}
