#include "ArtifactMoveRules.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

int main()
{
    using explorer_rl::backpackInsertionFits;
    // The exact failed one-item action is invalid, while an external append
    // into an empty or one-item destination must remain available.
    assert(!backpackInsertionFits(1, true, 1));
    assert(backpackInsertionFits(0, false, 0));
    assert(backpackInsertionFits(1, false, 1));
    assert(!backpackInsertionFits(0, true, 0));
    assert(!backpackInsertionFits(1, true, -1));
    for(int size = 1; size <= 64; ++size)
    {
        std::vector<int> original(size);
        std::iota(original.begin(), original.end(), 0);
        for(int source = 0; source < size; ++source)
            for(int destination = 0; destination <= size + 1; ++destination)
            {
                auto inventory = original;
                const int artifact = inventory.at(source);
                inventory.erase(inventory.begin() + source);
                const bool valid = backpackInsertionFits(size, true, destination);
                if(!valid)
                {
                    assert(destination > static_cast<int>(inventory.size()));
                    continue;
                }
                assert(destination <= static_cast<int>(inventory.size()));
                inventory.insert(inventory.begin() + destination, artifact);
                assert(inventory.at(destination) == artifact && inventory.size() == original.size());
                inventory.erase(inventory.begin() + destination);
                inventory.insert(inventory.begin() + source, artifact);
                assert(inventory == original);
            }
    }
    std::cout << "artifact insertion bounds and reorder roundtrips passed\n";
}
