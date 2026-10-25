#include "StdInc.h"
#include "ResourceUsage.h"
#include <filesystem>
#include <fstream>
#include <sstream>
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
JsonNode resourceSnapshot()
{
	JsonNode result;
	for(const auto * field : {"peak_resident_bytes", "resident_bytes", "open_descriptors", "threads"})
		result[field] = JsonNode();
	if(const auto peak = peakResidentBytes()) result["peak_resident_bytes"].Integer() = *peak;
#ifdef __linux__
	std::ifstream status("/proc/self/status");
	std::string line;
	while(std::getline(status, line))
	{
		std::istringstream fields(line);
		std::string key;
		uint64_t value;
		if(fields >> key >> value)
		{
			if(key == "VmRSS:") result["resident_bytes"].Integer() = value * 1024;
			if(key == "Threads:") result["threads"].Integer() = value;
		}
	}
	try
	{
		uint64_t descriptors = 0;
		for(const auto & entry : std::filesystem::directory_iterator("/proc/self/fd"))
		{
			(void)entry;
			++descriptors;
		}
		result["open_descriptors"].Integer() = descriptors;
	}
	catch(const std::filesystem::filesystem_error &) {} // optional telemetry
#endif
	return result;
}

}
