#include "StdInc.h"
#include "ResourceUsage.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <libproc.h>
#include <unistd.h>
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
#ifdef __APPLE__
	mach_task_basic_info_data_t info{};
	mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
	if(task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
		result["resident_bytes"].Integer() = static_cast<int64_t>(info.resident_size);
	thread_act_array_t threads = nullptr;
	mach_msg_type_number_t threadCount = 0;
	if(task_threads(mach_task_self(), &threads, &threadCount) == KERN_SUCCESS)
	{
		result["threads"].Integer() = threadCount;
		for(mach_msg_type_number_t index = 0; index < threadCount; ++index)
			mach_port_deallocate(mach_task_self(), threads[index]);
		vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), threadCount * sizeof(thread_act_t));
	}
	const int bytes = proc_pidinfo(getpid(), PROC_PIDLISTFDS, 0, nullptr, 0);
	if(bytes > 0)
	{
		std::vector<proc_fdinfo> descriptors(static_cast<size_t>(bytes) / sizeof(proc_fdinfo) + 16);
		const int used = proc_pidinfo(getpid(), PROC_PIDLISTFDS, 0, descriptors.data(), static_cast<int>(descriptors.size() * sizeof(proc_fdinfo)));
		if(used > 0)
			result["open_descriptors"].Integer() = used / static_cast<int>(sizeof(proc_fdinfo));
	}
#endif
	return result;
}

}
