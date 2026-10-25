#include "StdInc.h"
#include "ResourceUsage.h"
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
namespace native_session_host
{
std::optional<uint64_t> peakResidentBytes()
{
#if defined(__APPLE__) || defined(__linux__)
	rusage usage{};
	if(getrusage(RUSAGE_SELF, &usage) != 0) return std::nullopt;
#ifdef __APPLE__
	return static_cast<uint64_t>(usage.ru_maxrss);
#else
	return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
#else
	return std::nullopt;
#endif
}
}
