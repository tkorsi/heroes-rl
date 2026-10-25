// Actual contextual evidence ownership and real shared-library registry.
// No game, server, map, policy or simulated engine is constructed.
#include "StdInc.h"
#include "ContextualEvidence.h"

#include <cstdlib>
#include <iostream>

namespace
{
void check(bool value)
{
	if(!value)
		throw std::runtime_error("contextual evidence lifetime check failed");
}

std::shared_ptr<explorer_rl::ContextualEvidence> replayOwner()
{
	JsonNode header;
	header["kind"].String() = "init";
	header["schema"].String() = explorer_rl::ContextualEvidence::SCHEMA;
	header["episode"].String() = "lifetime-only";
	header["interface"].String() = "adventure-windows-v1";
	header["protocol_version"].Integer() = 4;
	header["history_schema"].String() = "heroes-player-history-v2";
	header["action_schema"].String() = "heroes-context-actions-v2";
	return std::make_shared<explorer_rl::ContextualEvidence>(std::vector<JsonNode>{header}, true);
}
}

int main(int argc, char ** argv)
{
	try
	{
		if(argc != 3)
			throw std::runtime_error("requires a lifecycle mode and temporary directory");
		const std::string mode = argv[1];
		if(mode == "live-exit" || mode == "session-close")
		{
			setenv("VCMI_EXPERIMENT_DIR", argv[2], 1);
			auto & owner = explorer_rl::ContextualEvidence::live("lifetime-only", false);
			check(explorer_rl::ContextualEvidence::current() == &owner);
			if(mode == "session-close")
			{
				auto retained = experimentEvidenceHooks();
				std::weak_ptr<ExperimentEvidenceHooks> lifetime = retained;
				setExperimentEvidenceHooks(nullptr);
				retained->finishNativeSession();
				retained.reset();
				check(lifetime.expired());
				resetNativeSessionEvidence();
				check(experimentPackPosition().index == 0);
			}
			// Both the live recorder and registry retain ownership until static
			// teardown, exactly as in a successfully terminated native process.
		}
		else
		{
			check(mode == "clear" || mode == "registry-exit");
			auto owner = replayOwner();
			std::weak_ptr<explorer_rl::ContextualEvidence> lifetime = owner;
			setExperimentEvidenceHooks(owner);
			owner.reset();
			check(!lifetime.expired());
			if(mode == "clear")
			{
				// Destruction occurs while the registry setter holds its mutex.
				// Looking up the registry from the destructor would deadlock.
				setExperimentEvidenceHooks(nullptr);
				check(lifetime.expired());
				check(explorer_rl::ContextualEvidence::current() == nullptr);
			}
			// registry-exit leaves the registry as the only strong owner. Its
			// final release must not resurrect that dying shared_ptr recursively.
		}
		std::cout << "contextual evidence lifetime: " << mode << " passed\n";
		return 0;
	}
	catch(const std::exception & error)
	{
		std::cerr << error.what() << '\n';
		return 1;
	}
}
