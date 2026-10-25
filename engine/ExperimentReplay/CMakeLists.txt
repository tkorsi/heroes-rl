# Replays an experiment's recorded server inputs and verifies its digests.
add_executable(vcmireplay StdInc.cpp StdInc.h ReplayMain.cpp ReplayWorker.cpp ReplayWorker.h ForkEpisode.h WorkerProtocol.cpp WorkerProtocol.h)
target_link_libraries(vcmireplay PRIVATE vcmi vcmiservercommon)
if(ENABLE_CLIENT)
	# Compile the exact adapter implementation into the offline verifier; no
	# dynamic AI loading or socket policy server is used by contextual replay.
	target_sources(vcmireplay PRIVATE ContextualReplay.cpp ContextualReplay.h ReplayClientHost.cpp
		../AI/ExplorerRL/ExplorerRL.cpp ../AI/ExplorerRL/Contextual.cpp
		../AI/ExplorerRL/ContextualEvidence.cpp ../AI/ExplorerRL/TownEconomy.cpp
		../AI/ExplorerRL/TownRecruitment.cpp ../AI/ExplorerRL/Tavern.cpp ../AI/ExplorerRL/ArmyOperations.cpp ../AI/ExplorerRL/Exchange.cpp ../AI/ExplorerRL/Dialog.cpp ../AI/ExplorerRL/DialogContext.cpp ../AI/ExplorerRL/TownTransfer.cpp ../AI/ExplorerRL/Notice.cpp
		../AI/ExplorerRL/KnowledgePlanner.cpp)
	target_link_libraries(vcmireplay PRIVATE vcmiclientcommon)
	target_compile_definitions(vcmireplay PRIVATE HEROES_CONTEXTUAL_REPLAY=1)
endif()
target_include_directories(vcmireplay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
vcmi_set_output_dir(vcmireplay "")
enable_pch(vcmireplay)
