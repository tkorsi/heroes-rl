// Production CClient::sendRequest with real mutexes and actual UnlockGuard.
// Transport/recording are fakes; no VCMI game or AI policy is executed.
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#define VCMI_LIB_NAMESPACE_BEGIN
#define VCMI_LIB_NAMESPACE_END
#include "vcmi/lib/UnlockGuard.h"

struct CGameState { inline static std::shared_mutex mutex; };
struct PlayerColor {
    int value = 0;
    PlayerColor() = default;
    explicit PlayerColor(int value) : value(value) {}
    auto operator<=>(const PlayerColor &) const = default;
};
struct CPackForServer {
    mutable int requestID = -1;
    mutable PlayerColor player;
    std::string payload = "owned-value";
};
std::recursive_mutex eventMutex;
thread_local unsigned eventDepth = 0;
int learner = 1;
bool expectShared = true, failEvent = false, failCapture = false;
std::atomic<int> captured{0}, sentCount{0}, notifications{0};
std::promise<int> sent;
int experimentBuiltinLearner() { return learner; }
bool contextualEvidenceEnabled() { return learner >= 0; }
std::string experimentEventActor() { return eventDepth ? "client" : ""; }

bool writerCanLock() {
    bool result = false;
    std::thread probe([&] {
        result = CGameState::mutex.try_lock();
        if(result) CGameState::mutex.unlock();
    });
    probe.join();
    return result;
}
void checkRequestScope() {
    assert(eventDepth > 0);
    assert(writerCanLock() == !expectShared);
}
struct ExperimentEventGuard {
    std::unique_lock<std::recursive_mutex> lock;
    ExperimentEventGuard(const char *, const char *) : lock(eventMutex) {
        if(failEvent) throw std::runtime_error("event constructor failure");
        ++eventDepth;
    }
    ~ExperimentEventGuard() { --eventDepth; }
};
struct Hooks {
    void nativeRequest(const CPackForServer & request, PlayerColor) {
        checkRequestScope();
        assert(request.payload == "owned-value");
        if(failCapture) throw std::runtime_error("capture failure");
        ++captured;
    }
};
std::shared_ptr<Hooks> experimentEvidenceHooks() { return std::make_shared<Hooks>(); }
struct Logger { template<class... Args> void trace(const char *, Args...) {} };
Logger logger;
Logger * logGlobal = &logger, * logNetwork = &logger;
struct Waiting {
    std::mutex mutex;
    std::condition_variable changed;
    std::set<int> ids;
    void pushBack(int value) { std::lock_guard lock(mutex); ids.insert(value); }
    void waitWhileContains(int value) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return !ids.contains(value); });
    }
    void acknowledge(int value) {
        std::lock_guard lock(mutex);
        assert(ids.erase(value) == 1);
        changed.notify_all();
    }
};
struct Interface {
    void requestSent(const CPackForServer *, int) { checkRequestScope(); ++notifications; }
};
struct Server {
    void sendGamePack(const CPackForServer & request) {
        checkRequestScope();
        ++sentCount;
        sent.set_value(request.requestID);
    }
};
struct Game { Server value; Server & server() { return value; } } game;
Game * GAME = &game;
namespace vstd {
template<class Map, class Key> bool contains(const Map & map, const Key & key) { return map.contains(key); }
}
struct CClient {
    std::atomic<int> requestCounter{1};
    Waiting waitingRequest;
    bool experimentReplayMode = false;
    std::map<PlayerColor, std::shared_ptr<Interface>> playerint;
    int sendRequest(const CPackForServer &, PlayerColor, bool);
};
// PRODUCTION_SEND_REQUEST

int main(int argc, char ** argv) {
    assert(argc == 2);
    const std::string mode = argv[1];
    CClient client;
    PlayerColor player(0);
    if(mode == "legacy") learner = -1;
    if(mode == "learner") player = PlayerColor(1);
    client.playerint[player] = std::make_shared<Interface>();
    CPackForServer request;
    if(mode == "event-failure" || mode == "capture-failure") {
        failEvent = mode == "event-failure";
        failCapture = mode == "capture-failure";
        std::shared_lock read(CGameState::mutex);
        bool failed = false;
        try { client.sendRequest(request, player, true); }
        catch(const std::runtime_error &) { failed = true; }
        assert(failed && !writerCanLock() && eventDepth == 0);
        assert(eventMutex.try_lock()); eventMutex.unlock();
        assert(sentCount == 0 && captured == 0 && notifications == 0);
        read.unlock();
        assert(writerCanLock());
    } else if(mode == "no-wait" || mode == "nested-no-wait") {
        expectShared = false;
        std::optional<ExperimentEventGuard> parent;
        if(mode == "nested-no-wait") parent.emplace("client", "packet");
        assert(client.sendRequest(request, player, false) == 1);
        parent.reset();
        assert(writerCanLock() && captured == 1 && sentCount == 1 && notifications == 1);
        assert(eventDepth == 0);
    } else {
        assert(mode == "contended" || mode == "legacy" || mode == "learner");
        std::shared_lock read(CGameState::mutex);
        std::promise<void> packetOwnsEvent;
        auto entered = packetOwnsEvent.get_future();
        auto delivered = sent.get_future();
        std::atomic<int> applied{0};
        std::thread packet([&] {
            if(mode == "contended") {
                std::unique_lock event(eventMutex);
                packetOwnsEvent.set_value();
                // Exactly the recorded inversion: packet owns event and waits GS
                // while the native request's caller already owns shared GS.
                std::unique_lock state(CGameState::mutex);
                ++applied;
            }
            const int id = delivered.get();
            std::unique_lock event(eventMutex);
            std::unique_lock state(CGameState::mutex);
            ++applied;
            client.waitingRequest.acknowledge(id);
        });
        if(mode == "contended") entered.get();
        assert(client.sendRequest(request, player, true) == 1);
        assert(!writerCanLock() && eventDepth == 0);
        read.unlock();
        packet.join();
        assert(writerCanLock());
        assert(applied == (mode == "contended" ? 2 : 1));
        assert(captured == (mode == "legacy" ? 0 : 1));
        assert(sentCount == 1 && notifications == 1);
    }
    std::cout << mode << " passed\n";
}
