// Exercises the production contextual capture tracker with synthetic worlds.
// The fixture owns hidden state; the tracker may only learn what tileVisible,
// tileFacts, visibleObjectsAt and ownState expose, exactly like the engine adapter.
#include "ContextualState.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <set>

using explorer_rl::ContextChange;
using explorer_rl::ContextOwn;
using explorer_rl::ContextSnapshot;
using explorer_rl::ContextTile;
using explorer_rl::ContextTracker;
using explorer_rl::VisibleObject;

namespace
{
int checks = 0;
int groups = 0;

void expect(bool condition, const char * message)
{
	++checks;
	if(!condition)
	{
		std::cerr << "FAILED: " << message << '\n';
		std::exit(1);
	}
}

struct World
{
	int sizeX = 6;
	int sizeY = 5;
	std::set<std::array<int, 3>> visible;
	std::map<std::array<int, 3>, int> roads; // hidden truth
	std::vector<VisibleObject> objects; // hidden truth
	ContextOwn own{{20, 5, 20, 5, 5, 5, 7500}, {{9, {1, 1, 0}, 1500, 1500}, {4, {3, 2, 0}, 1200, 1500}}, {{7, {0, 0, 0}}}};
	mutable int hiddenQueries = 0;

	int width() const { return sizeX; }
	int height() const { return sizeY; }
	int levels() const { return 1; }
	bool tileVisible(int x, int y, int z) const { return visible.contains({x, y, z}); }
	ContextOwn ownState() const { return own; }

	ContextTile tileFacts(int x, int y, int z) const
	{
		if(!tileVisible(x, y, z))
			++hiddenQueries;
		const auto road = roads.find({x, y, z});
		const bool occupied = std::any_of(objects.begin(), objects.end(),
			[x, y, z](const auto & object) { return object.x == x && object.y == y && object.z == z; });
		return {(x + y) % 10, road == roads.end() ? 0 : road->second, false, occupied};
	}

	void visibleObjectsAt(int x, int y, int z, std::vector<VisibleObject> & out) const
	{
		if(!tileVisible(x, y, z))
			++hiddenQueries;
		for(const auto & object : objects)
			if(object.x == x && object.y == y && object.z == z)
				out.push_back(object);
	}

	void see(int x0, int y0, int x1, int y1)
	{
		for(int y = y0; y <= y1; ++y)
			for(int x = x0; x <= x1; ++x)
				visible.insert({x, y, 0});
	}
};

/// The adapter's fold: the ordered change stream alone must rebuild each capture.
struct Fold
{
	std::map<std::array<int, 3>, ContextTile> tiles;
	std::map<int64_t, VisibleObject> objects;
	ContextOwn own;

	void apply(const ContextChange & change)
	{
		for(const auto & position : change.hidden)
			tiles.erase(position);
		for(const auto & fact : change.revealed)
			tiles[fact.position] = fact.tile;
		for(const auto & fact : change.tilesChanged)
			tiles[fact.position] = fact.tile;
		for(const auto id : change.disappeared)
			objects.erase(id);
		for(const auto & object : change.appeared)
			objects[object.id] = object;
		for(const auto & object : change.changed)
			objects[object.id] = object;
		if(change.own)
			own = *change.own;
	}

	bool matches(const ContextSnapshot & snapshot) const
	{
		std::map<std::array<int, 3>, ContextTile> expected;
		for(size_t index = 0; index < snapshot.tiles.size(); ++index)
			if(snapshot.tiles[index])
				expected[{static_cast<int>(index % snapshot.width), static_cast<int>(index / snapshot.width % snapshot.height),
					static_cast<int>(index / (static_cast<size_t>(snapshot.width) * snapshot.height))}] = snapshot.facts[index];
		return expected == tiles && snapshot.objects == objects && snapshot.own == own;
	}
};

void firstCaptureAndOrdering()
{
	++groups;
	World world;
	world.see(0, 0, 2, 2);
	world.roads[{1, 0, 0}] = 2;
	world.objects.push_back({11, 2, 2, 0, 79, 6, 255});
	world.objects.push_back({12, 5, 4, 0, 54, 1, 255}); // hidden
	ContextTracker tracker;
	const auto first = tracker.update(explorer_rl::captureContext(world));
	expect(first.revealed.size() == 9, "first capture reveals every visible tile");
	expect(first.appeared.size() == 1 && first.appeared.front().id == 11, "first capture lists only visible objects");
	expect(first.own.has_value(), "first capture carries the own state");
	expect(first.own->heroes.size() == 2 && first.own->heroes[0].id == 4 && first.own->heroes[1].id == 9, "own heroes ascend by ID");
	const auto road = std::find_if(first.revealed.begin(), first.revealed.end(),
		[](const auto & fact) { return fact.position == std::array<int, 3>{1, 0, 0}; });
	expect(road != first.revealed.end() && road->tile.road == 2, "a revealed tile carries its road");
	expect(world.hiddenQueries == 0, "hidden tiles are never queried");
	expect(tracker.update(explorer_rl::captureContext(world)).empty(), "an unchanged world reports nothing");
}

void roadsOwnStateAndLosses()
{
	++groups;
	World world;
	world.see(0, 0, 3, 3);
	ContextTracker tracker;
	tracker.update(explorer_rl::captureContext(world));
	const auto digest = explorer_rl::contextDigest(*tracker.latest());
	world.roads[{2, 2, 0}] = 1;
	const auto paved = tracker.update(explorer_rl::captureContext(world));
	expect(paved.tilesChanged.size() == 1 && paved.tilesChanged.front().tile.road == 1 && !paved.own, "a road change is a tile change");
	expect(explorer_rl::contextDigest(*tracker.latest()) != digest, "roads participate in the digest");
	world.own.heroes[0].position = {2, 1, 0};
	world.own.heroes[0].movement = 1400;
	const auto moved = tracker.update(explorer_rl::captureContext(world));
	expect(moved.own && moved.own->heroes.size() == 2 && moved.heroesLost.empty(), "a move reports the complete own state");
	world.own.heroes.erase(world.own.heroes.begin());
	const auto lost = tracker.update(explorer_rl::captureContext(world));
	expect(lost.heroesLost.size() == 1 && lost.heroesLost.front() == 9, "a hero no longer owned is lost by ID");
	expect(tracker.update(explorer_rl::captureContext(world)).empty(), "a loss is reported once");
	world.visible.erase({3, 3, 0});
	const auto hidden = tracker.update(explorer_rl::captureContext(world));
	expect(hidden.hidden.size() == 1 && hidden.revealed.empty() && hidden.tilesChanged.empty(), "a hidden tile carries no facts");
	World resized = world;
	resized.sizeX = 7;
	bool threw = false;
	try
	{
		tracker.update(explorer_rl::captureContext(resized));
	}
	catch(const std::runtime_error &)
	{
		threw = true;
	}
	expect(threw, "map dimensions cannot change during an episode");
}

void randomFold()
{
	++groups;
	std::mt19937 random(20261024);
	World world;
	world.sizeX = 9;
	world.sizeY = 7;
	ContextTracker tracker;
	Fold fold;
	for(int step = 0; step < 400; ++step)
	{
		const auto pick = [&](int bound) { return static_cast<int>(random() % bound); };
		switch(pick(6))
		{
			case 0:
				world.see(pick(9), pick(7), pick(9), pick(7));
				break;
			case 1:
				if(!world.visible.empty())
					world.visible.erase(std::next(world.visible.begin(), pick(static_cast<int>(world.visible.size()))));
				break;
			case 2:
				world.roads[{pick(9), pick(7), 0}] = pick(4);
				break;
			case 3:
				world.objects.push_back({100 + step, pick(9), pick(7), 0, pick(100), pick(8), pick(2) ? 255 : 1});
				break;
			case 4:
				if(!world.objects.empty())
					world.objects.erase(world.objects.begin() + pick(static_cast<int>(world.objects.size())));
				break;
			default:
				if(!world.own.heroes.empty())
				{
					auto & hero = world.own.heroes[pick(static_cast<int>(world.own.heroes.size()))];
					hero.position = {pick(9), pick(7), 0};
					hero.movement = pick(1500);
				}
				world.own.resources[pick(7)] += pick(3);
		}
		fold.apply(tracker.update(explorer_rl::captureContext(world)));
		expect(fold.matches(*tracker.latest()), "the change stream folds into every capture");
	}
	expect(world.hiddenQueries == 0, "hidden tiles are never queried while folding");
}
}

int main()
{
	firstCaptureAndOrdering();
	roadsOwnStateAndLosses();
	randomFold();
	std::cout << "context state: " << groups << " groups, " << checks << " checks passed\n";
	return 0;
}
