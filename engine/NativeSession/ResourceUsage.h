#pragma once
#include <cstdint>
#include <optional>
namespace native_session_host
{
/// Process high-water mark, cumulative across reused episodes, not a per-game delta.
std::optional<uint64_t> peakResidentBytes();
}
