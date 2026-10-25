#include "StdInc.h"

#include "../client/CMT.h"
#include "../lib/callback/CGameInterface.h"

/// Offline replay has no application event loop or GUI. Reaching one of the
/// interactive executable's termination hooks is a verification failure;
/// the verified barrier is handled synchronously by the replay coordinator.
void requestHeadlessShutdown(int exitCode)
{
	const auto message = "Unexpected client shutdown during contextual replay: " + std::to_string(exitCode);
	reportExperimentFailure(message);
	throw std::runtime_error(message);
}

bool headlessShutdownRequested()
{
	return false;
}

[[noreturn]] void handleFatalError(const std::string & message, bool)
{
	reportExperimentFailure("Fatal client error during contextual replay: " + message);
	throw std::runtime_error(message);
}
