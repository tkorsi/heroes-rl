#include "StdInc.h"
#include "RuntimeEnvironment.h"
#include <cstdlib>
#include <stdexcept>
namespace native_session_host
{
namespace
{
void change(const std::string & key, const std::optional<std::string> & value)
{
#ifdef _WIN32
	if(_putenv_s(key.c_str(), value ? value->c_str() : "") != 0)
#else
	if((value ? ::setenv(key.c_str(), value->c_str(), 1) : ::unsetenv(key.c_str())) != 0)
#endif
		throw std::runtime_error("Cannot set native runtime selector: " + key);
}
}
void RuntimeEnvironment::set(const std::string & key, const std::optional<std::string> & value)
{
	if(!original.contains(key))
	{
		const char * old = std::getenv(key.c_str());
		original.emplace(key, old ? std::optional<std::string>(old) : std::nullopt);
	}
	change(key, value);
}
RuntimeEnvironment::~RuntimeEnvironment()
{
	for(const auto & [key, value] : original)
		try { change(key, value); } catch(...) {}
}
}
