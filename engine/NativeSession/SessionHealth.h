/* Native session failure admission. GPL-2.0-or-later. */
#pragma once
#include <optional>
#include <stdexcept>
#include <utility>

namespace native_session_host
{
/// Controller-thread state. Once a native episode fails, this owner is never
/// reusable, including when the failure is discovered while joining its threads.
class SessionHealth
{
	bool poisoned = false;
public:
	void poison() noexcept { poisoned = true; }
	bool failed() const noexcept { return poisoned; }
	void requireResetReady(bool loaded) const
	{
		if(!loaded || poisoned)
			throw std::logic_error("Load resources first; failed native sessions require a fresh process");
	}
	template<class Cleanup> void prepareReset(bool loaded, Cleanup && cleanup)
	{
		requireResetReady(loaded);
		try { std::forward<Cleanup>(cleanup)(); }
		catch(...) { poison(); throw; }
		// Cleanup joins producers and may discover a failure from the old game.
		// No new-episode mutation is permitted before this second admission.
		requireResetReady(loaded);
	}
};

/// Protected by the owning Shutdown mutex. A successful cleanup notification
/// is not a new outcome, but a nonzero report must survive even after stop starts.
struct ShutdownStatus
{
	std::optional<int> code;
	bool stopping = false;
	bool record(int value) noexcept
	{
		if(stopping && value == 0) return false;
		if(!code || value != 0) code = value;
		return true;
	}
	bool failed() const noexcept { return code.has_value() && *code != 0; }
};
}
