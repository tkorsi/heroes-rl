#pragma once
#include <cstdint>
#include <optional>
#include "../lib/json/JsonNode.h"
namespace native_session_host
{
/// Process high-water mark, cumulative across reused episodes, not a per-game delta.
std::optional<uint64_t> peakResidentBytes();
/// Optional host telemetry only. Unsupported/unreadable fields are null; /proc
/// is never required for game execution or cleanup. No input/provenance claim.
JsonNode resourceSnapshot();
}
