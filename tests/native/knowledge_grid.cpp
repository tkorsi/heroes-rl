// Exercises the production knowledge grid the contextual route planner reads.
// A synthetic world owns the hidden truth; the grid is filled only from the
// tiles one team has explored, exactly as KnowledgePlanner fills it from the
// player's callback, and must give VCMI 1.7.5's guard and visit answers.
#include "KnowledgeGrid.h"

#include <cstdlib>
#include <iostream>
#include <random>
#include <set>

using explorer_rl::GridPosition;
using explorer_rl::KnowledgeGrid;
using explorer_rl::KnownObject;
using explorer_rl::KnownTile;
using explorer_rl::PlannedNode;

namespace
{
[[noreturn]] void fail(const char * message)
{
	std::cerr << "FAILED: " << message << '\n';
	std::exit(1);
}

void expect(bool condition, const char * message)
{
	if(!condition)
		fail(message);
}

constexpr uint8_t ALL_DIRECTIONS = 0xff;

KnownObject monster(int64_t id)
{
	// Wandering monsters block their tile and are visited from every side.
	return {id, true, true, true, ALL_DIRECTIONS};
}

/// The hidden truth: every tile's terrain and objects, and the team's fog.
struct World
{
	int width = 7;
	int height = 6;
	std::vector<KnownTile> truth = std::vector<KnownTile>(static_cast<size_t>(width) * height,
		KnownTile{true, false, true, {}});
	std::set<std::pair<int, int>> explored;

	KnownTile & tile(int x, int y) { return truth[static_cast<size_t>(x) + width * y]; }

	/// What the planner may know: explored tiles only, the rest stays at the
	/// grid's defaults. This mirrors KnowledgeMap::capture.
	KnowledgeGrid knowledge() const
	{
		KnowledgeGrid grid(width, height, 1);
		for(const auto & [x, y] : explored)
			grid.at({x, y, 0}) = truth[static_cast<size_t>(x) + width * y];
		return grid;
	}

	void exploreAll()
	{
		for(int y = 0; y < height; ++y)
			for(int x = 0; x < width; ++x)
				explored.insert({x, y});
	}
};

void testVisitDirections()
{
	// ObjectTemplate::isVisitableFrom bit layout: 1 2 4 / 128 _ 8 / 64 32 16.
	const std::array<std::array<int, 3>, 8> bits{{
		{{-1, -1, 1}}, {{0, -1, 2}}, {{1, -1, 4}}, {{1, 0, 8}},
		{{1, 1, 16}}, {{0, 1, 32}}, {{-1, 1, 64}}, {{-1, 0, 128}}
	}};
	for(const auto & [dx, dy, bit] : bits)
		for(const auto & [ox, oy, other] : bits)
			expect(KnowledgeGrid::visitableFromDirection(static_cast<uint8_t>(bit), ox, oy) == (bit == other),
				"A visit direction bit must open exactly its own neighbour");
	expect(KnowledgeGrid::visitableFromDirection(0, 0, 0), "An object's own tile is always reachable from itself");
	expect(KnowledgeGrid::visitableFromDirection(2, 0, -5), "Offsets are compared by sign, as in the engine");
}

void testUnknownTilesNeverGuardOrBlock()
{
	World world;
	world.exploreAll();
	world.explored.erase({4, 2});
	world.tile(4, 2).visitable.push_back(monster(7)); // hidden monster next to (3, 2)
	const auto grid = world.knowledge();
	expect(!grid.guard({3, 2, 0}), "A monster on an unexplored tile must not guard a known tile");
	expect(grid.guardingCreatures({3, 2, 0}).empty(), "A hidden monster must not be counted");
	expect(!grid.guard({4, 2, 0}), "An unexplored tile has no guard");
	expect(!grid.visitableFrom({3, 2, 0}, {4, 2, 0}), "An unexplored tile is impassable rock");
	expect(grid.visitableFrom({3, 2, 0}, {2, 2, 0}), "An empty known tile can be entered");

	// The same monster, once its tile is explored, guards its neighbours.
	world.explored.insert({4, 2});
	const auto revealed = world.knowledge();
	expect(revealed.guard({3, 2, 0}) == GridPosition({4, 2, 0}), "A visible monster guards its neighbourhood");
	expect(revealed.guard({4, 2, 0}) == GridPosition({4, 2, 0}), "A monster's own tile is guarded by it first");
	// The engine counts a monster standing on the asked tile twice.
	expect(revealed.guardingCreatures({4, 2, 0}) == std::vector<int64_t>({7, 7}), "Guard count must keep the engine's double count");
	expect(revealed.guardingCreatures({3, 2, 0}) == std::vector<int64_t>({7}), "A neighbouring monster counts once");
}

void testGuardRules()
{
	World world;
	world.exploreAll();
	// A monster on water does not guard land, and the other way round.
	world.tile(1, 1).water = true;
	world.tile(1, 1).visitable.push_back(monster(1));
	auto grid = world.knowledge();
	expect(!grid.guard({2, 1, 0}), "A water monster must not guard land");
	world.tile(2, 1).water = true;
	grid = world.knowledge();
	expect(grid.guard({2, 1, 0}) == GridPosition({1, 1, 0}), "A water monster guards water");

	// A blocking visitable object on the asked tile, closed towards the
	// monster, keeps that monster from attacking it.
	World closed;
	closed.exploreAll();
	closed.tile(3, 3).visitable.push_back(monster(2));
	closed.tile(3, 2).visitable.push_back({3, false, true, true, 32}); // visitable only from the south
	grid = closed.knowledge();
	expect(grid.guard({3, 2, 0}) == GridPosition({3, 3, 0}), "An object open to the monster's side is attacked");
	closed.tile(3, 2).visitable.back().visitDir = 2; // only from the north
	grid = closed.knowledge();
	expect(!grid.guard({3, 2, 0}), "An object closed to the monster's side is not attacked");
	expect(!grid.visitableFrom({3, 3, 0}, {3, 2, 0}), "Visit direction applies to steps too");
	expect(grid.visitableFrom({3, 1, 0}, {3, 2, 0}), "An open side admits the step");
	// A visitable object that does not block ignores direction.
	closed.tile(3, 2).visitable.back().blockingHere = false;
	grid = closed.knowledge();
	expect(grid.guard({3, 2, 0}) == GridPosition({3, 3, 0}), "A non-blocking object never shields its tile");

	// The neighbourhood is scanned column by column from the north-west.
	World order;
	order.exploreAll();
	order.tile(3, 1).visitable.push_back(monster(4)); // north
	order.tile(2, 3).visitable.push_back(monster(5)); // south-west, in the first column
	grid = order.knowledge();
	expect(grid.guard({3, 2, 0}) == GridPosition({2, 3, 0}), "The first column scanned wins");
	expect(grid.guardingCreatures({3, 2, 0}) == std::vector<int64_t>({5, 4}), "Guards are listed in scan order");
	expect(grid.guard({0, 0, 0}) == std::nullopt, "A corner outside every zone has no guard");
}

/// Two worlds that differ only on unexplored tiles must give identical
/// answers everywhere, whatever the hidden terrain and objects are.
void testHiddenVariantsAreIndistinguishable()
{
	std::mt19937 random(20261024);
	for(int fold = 0; fold < 200; ++fold)
	{
		World first;
		for(int y = 0; y < first.height; ++y)
			for(int x = 0; x < first.width; ++x)
			{
				auto & tile = first.tile(x, y);
				tile.water = random() % 5 == 0;
				tile.entrable = random() % 7 != 0;
				if(random() % 4 == 0)
					tile.visitable.push_back(random() % 2 ? monster(x + 10 * y)
						: KnownObject{x + 10 * y, false, random() % 2 == 0, random() % 2 == 0, static_cast<uint8_t>(random())});
				if(random() % 3 != 0)
					first.explored.insert({x, y});
			}
		World second = first;
		for(int y = 0; y < second.height; ++y)
			for(int x = 0; x < second.width; ++x)
				if(!second.explored.contains({x, y}))
				{
					auto & tile = second.tile(x, y);
					tile.water = !tile.water;
					tile.entrable = !tile.entrable;
					tile.visitable.clear();
					if(random() % 2 == 0)
						tile.visitable.push_back(monster(1000 + x + 10 * y));
				}
		const auto a = first.knowledge();
		const auto b = second.knowledge();
		for(int y = 0; y < first.height; ++y)
			for(int x = 0; x < first.width; ++x)
			{
				const GridPosition position{x, y, 0};
				expect(a.guard(position) == b.guard(position), "A hidden difference changed a guard");
				expect(a.guardingCreatures(position) == b.guardingCreatures(position), "A hidden difference changed a guard count");
				for(int dy = -1; dy <= 1; ++dy)
					for(int dx = -1; dx <= 1; ++dx)
					{
						const GridPosition source{x + dx, y + dy, 0};
						if(a.inMap(source))
							expect(a.visitableFrom(source, position) == b.visitableFrom(source, position),
								"A hidden difference changed a step");
					}
			}
	}
}

void testCurrentTurnPrefix()
{
	using explorer_rl::currentTurnPrefix;
	// Destination first, origin last, as CPathsInfo::getPath lists them.
	const std::vector<PlannedNode> sameTurn{{{3, 0, 0}, 0}, {{2, 0, 0}, 0}, {{1, 0, 0}, 0}, {{0, 0, 0}, 0}};
	expect(currentTurnPrefix(sameTurn) == std::vector<GridPosition>({{1, 0, 0}, {2, 0, 0}, {3, 0, 0}}),
		"A same-turn route is walked whole, origin excluded");
	const std::vector<PlannedNode> later{{{3, 0, 0}, 2}, {{2, 0, 0}, 1}, {{1, 0, 0}, 0}, {{0, 0, 0}, 0}};
	expect(currentTurnPrefix(later) == std::vector<GridPosition>({{1, 0, 0}}), "The prefix stops before the next turn");
	const std::vector<PlannedNode> unaffordable{{{1, 0, 0}, 1}, {{0, 0, 0}, 0}};
	expect(currentTurnPrefix(unaffordable).empty(), "An unaffordable first step leaves no prefix");
	expect(currentTurnPrefix({{{0, 0, 0}, 0}}).empty(), "A lone origin has no steps");
	expect(currentTurnPrefix({}).empty(), "An empty path has no steps");
}
}

int main()
{
	testVisitDirections();
	testUnknownTilesNeverGuardOrBlock();
	testGuardRules();
	testHiddenVariantsAreIndistinguishable();
	testCurrentTurnPrefix();
	std::cout << "knowledge grid: 5 groups, 200 hidden-variant folds passed\n";
	return 0;
}
