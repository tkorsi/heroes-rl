// Harness for production spell target functions, compiled without a game runtime.
// Geometry/filter collaborators are explicit fakes; the selected production code
// is injected by test_spell_target_order.py. No game or policy is executed.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <vector>
using namespace std::placeholders;
struct Hex { int value = -1; bool isValid() const { return value >= 0; } };
namespace battle {
struct Unit {
    uint32_t id = 0;
    int tile = 0;
    bool valid = true, immune = false, invincible = false, living = true;
    uint32_t unitId() const { return id; }
    bool coversPos(Hex hex) const { return tile == hex.value; }
    bool alive() const { return living; }
    bool isInvincible() const { return invincible; }
    int getCount() const { return 20; }
    int getMaxHealth() const { return 10; }
    const Unit * acquireState() const { return this; }
};
using Units = std::vector<const Unit *>;
}
struct BattleStackAttacked {
    uint32_t stackAttacked = 0, attackerID = 0, killedAmount = 0;
    int64_t damageAmount = 0;
};
struct CStack : battle::Unit {
    // State mutation/RNG are outside this ordering regression. The production
    // Damage::apply chooses targets and calculated damage, then calls this fake.
    static void prepareAttacked(BattleStackAttacked & attack, int &, const battle::Unit * state) {
        assert(attack.stackAttacked == state->unitId());
        attack.killedAmount = std::min<int64_t>(state->getCount(), attack.damageAmount / state->getMaxHealth());
    }
};
struct StacksInjured { int battleID = -1; std::vector<BattleStackAttacked> stacks; };
struct BattleLogMessage { int battleID = -1; std::vector<int> lines; };
struct ServerCallback {
    int rng = 0;
    std::vector<StacksInjured> packets;
    bool describeChanges() const { return false; }
    int * getRNG() { return &rng; }
    void apply(const StacksInjured & packet) { packets.push_back(packet); }
    void apply(const BattleLogMessage &) { std::abort(); }
};
struct Destination {
    const battle::Unit * unitValue = nullptr;
    Hex hexValue;
    Destination() = default;
    explicit Destination(const battle::Unit * value) : unitValue(value), hexValue{value->tile} {}
    explicit Destination(Hex value) : hexValue(value) {}
};
using Target = std::vector<Destination>;
using EffectTarget = Target;
struct Battle {
    std::vector<const CStack *> units;
    const Battle * getBattle() const { return this; }
    int getBattleID() const { return 7; }
    template<class Predicate> battle::Units battleGetUnitsIf(Predicate predicate) const {
        battle::Units result;
        for(const auto * unit : units) if(predicate(unit)) result.push_back(unit);
        return result;
    }
    const CStack * battleGetStackByID(uint32_t id, bool) const {
        for(const auto * unit : units) if(unit->unitId() == id) return unit;
        std::abort();
    }
};
struct Mechanics {
    Battle * callback = nullptr;
    bool massive = false, alwaysFirst = false;
    const Battle * battle() const { return callback; }
    bool isMassive() const { return massive; }
    bool alwaysHitFirstTarget() const { return alwaysFirst; }
    int getEffectValue() const { return 37; }
    int64_t adjustEffectValue(const battle::Unit * unit) const { return 40 + unit->unitId(); }
};
struct Logger { void debug(const char *) const {} };
const Logger logger;
const Logger * logGlobal = &logger;
namespace vstd {
template<class Container> void concatenate(Container & left, const Container & right) { left.insert(left.end(), right.begin(), right.end()); }
}
namespace effects {
struct Effect {
    Target selected;
    EffectTarget transformTarget(const Mechanics *, const Target &, const Target &) const { return selected; }
};
}
struct EffectList {
    std::vector<effects::Effect> entries;
    template<class Function> void forEachEffect(int, Function function) const {
        bool stop = false;
        for(const auto & effect : entries) function(&effect, stop);
    }
};
struct UnitEffect {
    bool getStackFilter(const Mechanics *, bool, const battle::Unit * unit) const { return unit->valid; }
    bool eraseByImmunityFilter(const Mechanics *, const battle::Unit * unit) const { return unit->immune; }
    EffectTarget transformTargetByRange(const Mechanics *, const Target &, const Target &) const;
};
struct Damage {
    bool killByPercentage = false, killByCount = false;
    int chainLength = 1;
    double chainFactor = .5;
    int64_t damageForTarget(size_t, const Mechanics *, const battle::Unit *) const;
    void apply(ServerCallback *, const Mechanics *, const EffectTarget &) const;
    void describeEffect(std::vector<int> &, const Mechanics *, const battle::Unit *, uint32_t, int64_t, bool) const {}
};
struct BattleSpellMechanics : Mechanics {
    std::shared_ptr<EffectList> effects = std::make_shared<EffectList>();
    std::vector<std::pair<int, EffectTarget>> effectsToApply;
    Target transformSpellTarget(const Target & value) const { return value; }
    int getEffectLevel() const { return 0; }
    battle::Units collectTargets() const;
    std::vector<const CStack *> getAffectedStacks(const Target &) const;
};
// PRODUCTION_FUNCTIONS

template<class Container> std::vector<uint32_t> ids(const Container & units) {
    std::vector<uint32_t> result;
    for(const auto * unit : units) result.push_back(unit->unitId());
    return result;
}
std::vector<const battle::Unit *> units(const Target & target) {
    std::vector<const battle::Unit *> result;
    for(const auto & destination : target) result.push_back(destination.unitValue);
    return result;
}
int main() {
    // Assign IDs to different addresses, and permute the delivered targets
    // independently: both heap layout and duplicate input order must disappear.
    std::array<uint32_t, 4> allocation{2, 7, 11, 23};
    size_t cases = 0;
    do {
        std::array<CStack, 4> storage;
        Battle callback;
        for(size_t index = 0; index < allocation.size(); ++index) {
            auto & stack = storage[index];
            stack.id = allocation[index];
            stack.tile = static_cast<int>(allocation[index]);
            stack.immune = stack.invincible = stack.id == 23;
            callback.units.push_back(&stack);
        }
        std::array<uint32_t, 4> delivery{2, 7, 11, 23};
        do {
            Target target;
            for(auto id : delivery) {
                auto * stack = callback.battleGetStackByID(id, false);
                target.emplace_back(stack);
                target.emplace_back(Hex{stack->tile});
                target.emplace_back(stack);
            }
            target.emplace_back(); // Invalid destinations must remain ignored.
            UnitEffect effect;
            Mechanics mechanics{&callback};
            const auto expected = std::vector<uint32_t>{2, 7, 11};
            for(bool massive : {false, true}) {
                mechanics.massive = massive;
                auto selected = effect.transformTargetByRange(&mechanics, {}, target);
                assert(ids(units(selected)) == expected);
                Damage damage;
                std::vector<int64_t> amounts;
                for(size_t index = 0; index < selected.size(); ++index)
                    amounts.push_back(damage.damageForTarget(index, &mechanics, selected[index].unitValue));
                assert((amounts == std::vector<int64_t>{42, 47, 51}));
                ServerCallback server;
                damage.apply(&server, &mechanics, selected);
                assert(server.packets.size() == 1);
                const auto & packet = server.packets.front();
                assert(packet.battleID == 7 && packet.stacks.size() == 3);
                for(size_t index = 0; index < packet.stacks.size(); ++index) {
                    const auto & attack = packet.stacks[index];
                    assert(attack.stackAttacked == expected[index]);
                    assert(attack.attackerID == uint32_t(-1));
                    assert(attack.damageAmount == amounts[index]);
                }
                // Native percent/count formulas remain selected-unit dependent.
                damage.killByPercentage = true;
                assert(damage.damageForTarget(0, &mechanics, selected[0].unitValue) == 70);
                damage.killByPercentage = false;
                damage.killByCount = true;
                assert(damage.damageForTarget(0, &mechanics, selected[0].unitValue) == 370);
            }
            mechanics.massive = false;
            auto first = callback.battleGetStackByID(7, false);
            auto single = effect.transformTargetByRange(&mechanics, {Destination(first)}, {});
            assert((ids(units(single)) == std::vector<uint32_t>{7}));
            // The native always-hit-first exception is preserved after immunity.
            mechanics.alwaysFirst = true;
            auto immune = callback.battleGetStackByID(23, false);
            auto forced = effect.transformTargetByRange(&mechanics, {Destination(immune)}, target);
            assert((ids(units(forced)) == std::vector<uint32_t>{2, 7, 11, 23}));
            BattleSpellMechanics spell;
            spell.callback = &callback;
            spell.effectsToApply = {{0, target}, {1, target}};
            assert((ids(spell.collectTargets()) == std::vector<uint32_t>{2, 7, 11, 23}));
            spell.effects->entries = {{target}, {target}};
            assert(ids(spell.getAffectedStacks({})) == expected);
            ++cases;
        } while(std::next_permutation(delivery.begin(), delivery.end()));
    } while(std::next_permutation(allocation.begin(), allocation.end()));
    assert(cases == 576);
    std::cout << cases << " allocation/input permutations verified\n";
}
