#pragma once

#include "../../lib/callback/ExperimentHooks.h"
#include "../../lib/json/JsonNode.h"

#include <condition_variable>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>

namespace explorer_rl
{
class ContextualPrefixFinished final : public std::exception
{
public:
	const char * what() const noexcept override { return "Recorded contextual cutoff prefix is complete"; }
};

/// Diagnostics-only capture and deterministic replay. No field from this
/// ledger is part of the player observation or recurrent model input.
class ContextualEvidence final : public ExperimentEvidenceHooks
{
public:
	static constexpr const char * SCHEMA = "heroes-contextual-replay-v1";
	static constexpr const char * BUILTIN_SCHEMA = "heroes-contextual-builtin-replay-v1";
	static ContextualEvidence & live(const std::string & episode, bool town, bool armies = false, bool heroes = false, bool management = false, bool dialogs = false, bool expanded = false);
	static ContextualEvidence * current();
	static bool replaying();
	static bool cancelled();

	explicit ContextualEvidence(const std::vector<JsonNode> & tape, bool prefix);
	~ContextualEvidence() override;
	void finishNativeSession() override;
	const JsonNode & header() const { return metadata; }
	void setAutomaticRequest(std::function<void(const JsonNode &)> callback);
	void beforeEvent(const std::string & actor, const std::string & operation) override;
	void eventEntered(const std::string & actor, const std::string & operation) override;
	void eventLeaving(const std::string & actor, const std::string & operation) override;
	void nativeRequest(const CPackForServer & pack, PlayerColor player) override;
	void callbacksComplete(const CPackForClient & pack, ExperimentReceiptPosition position) override;
	void terminalBarrierComplete(const ExperimentTerminalBarrier & pack) override;
	void message(int player, JsonNode & value); ///< value is unchanged on return or throw
	void message(int player, const JsonNode & value); ///< records a copy of value
	void response(int player, const JsonNode & value);
	JsonNode replayResponse(int player);
	/// Only the coordinator consumes automatic client requests or supplies
	/// regenerated packets. Workers consume their own transaction tokens.
	std::optional<std::pair<std::string, std::string>> nextTransaction();
	void automaticTransaction();
	void waitForProgress();
	void stop();
	void check() const;
	JsonNode summary(uint64_t regeneratedPackets) const;

private:
	ContextualEvidence(const std::string & episode, bool town, bool armies = false, bool heroes = false, bool management = false, bool dialogs = false, bool expanded = false);
	JsonNode metadata;
	std::ofstream output;
	std::vector<JsonNode> records;
	size_t cursor = 0;
	uint64_t nextSequence = 0;
	bool replay = false;
	bool builtin = false;
	bool prefix = false;
	bool stopping = false;
	bool sealed = false;
	bool openFinalTransaction = false;
	std::string error;
	std::optional<JsonNode> pendingClose;
	JsonNode actualTerminalBarrier;
	std::function<void(const JsonNode &)> automaticRequest;
	mutable std::mutex mutex;
	std::condition_variable changed;
	JsonNode loaded;
	JsonNode checked;
	void rejectSealedEvent(const std::string & actor, const std::string & operation) const;
	void append(JsonNode & record); ///< sets the record's seq
	void expect(JsonNode & record); ///< sets the record's seq
	void drainAutomatic();
	void advance();
	void fail(const std::string & message);
	void count(JsonNode & counts, const JsonNode & record) const;
};
}
