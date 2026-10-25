#include "TransferRules.h"

#include <cassert>
#include <iostream>

using namespace explorer_rl;

template<class F> void rejected(F operation)
{
	bool failed = false;
	try { operation(); } catch(const std::exception &) { failed = true; }
	assert(failed);
}

int main()
{
	TransferArmy source{}, destination{};
	source[2] = {0, 14};
	destination[0] = {0, 25};
	auto partial = transferEffect(source, destination, 2, 4, 6, false);
	assert((partial.source[2] == TransferStack{0, 8}));
	assert((partial.destination[4] == TransferStack{0, 6}));
	assert(partial.destination[0] == destination[0]);
	assert(partial.destinationTotal == 6);
	auto merge = transferEffect(partial.source, partial.destination, 2, 0, 8, false);
	assert((merge.source[2] == TransferStack{-1, 0}));
	assert((merge.destination[0] == TransferStack{0, 33}));
	assert(merge.destination[4] == partial.destination[4]);
	assert(merge.destinationTotal == 33); // native value must include the old 25
	assert(source[2].count == 14 && destination[0].count == 25); // immutable snapshots

	for(int slot : {-1, 7})
	{
		rejected([&]{ transferEffect(source, destination, slot, 0, 1, false); });
		rejected([&]{ transferEffect(source, destination, 2, slot, 1, false); });
	}
	for(int count : {-1, 0, 15}) rejected([&]{ transferEffect(source, destination, 2, 0, count, false); });
	rejected([&]{ transferEffect(source, destination, 1, 0, 1, false); });
	destination[1] = {1, 1};
	rejected([&]{ transferEffect(source, destination, 2, 1, 1, false); }); // no implicit swap
	for(int i = 0; i < 7; ++i) destination[i] = {i + 1, 1};
	rejected([&]{ transferEffect(source, destination, 2, 6, 1, false); }); // no auto capacity merge
	destination[6] = {0, 10};
	assert(transferEffect(source, destination, 2, 6, 14, false).destinationTotal == 24);

	rejected([&]{ transferEffect(source, destination, 2, 6, 14, true); });
	assert(transferEffect(source, destination, 2, 6, 13, true).source[2].count == 1);
	source[3] = {2, 1};
	assert(transferEffect(source, destination, 2, 6, 14, true).source[3].count == 1);
	source[3] = {};
	destination[6] = {0, std::numeric_limits<int32_t>::max() - 13};
	rejected([&]{ transferEffect(source, destination, 2, 6, 1, false); }); // native total overflows even for one
	destination[6].count -= 1;
	assert(transferEffect(source, destination, 2, 6, 14, false).destinationTotal == std::numeric_limits<int32_t>::max());
	source[4] = {0, 0};
	rejected([&]{ transferEffect(source, destination, 2, 6, 1, false); }); // no malformed empty stack
	std::cout << "transfer rules: exact split, merge, capacity, minimum army and int32 boundaries passed\n";
}
