#pragma once
#include <map>
#include <optional>
#include <string>
namespace native_session_host
{
/// Private host glue. Process-wide environment mutation is allowed only while
/// EngineSession owns the process and no previous episode threads remain.
class RuntimeEnvironment
{
	std::map<std::string, std::optional<std::string>> original;
public:
	~RuntimeEnvironment();
	void set(const std::string & key, const std::optional<std::string> & value);
};
}
