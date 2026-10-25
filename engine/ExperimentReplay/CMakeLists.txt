# Replays an experiment's recorded server inputs and verifies its digests.
add_executable(vcmireplay StdInc.cpp StdInc.h ReplayMain.cpp)
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

if(ENABLE_CLIENT)
	# Asset-free regression using actual client/network shutdown and TBB tasks.
	add_executable(native_headless_shutdown_order EXCLUDE_FROM_ALL tests/HeadlessShutdownOrder.cpp)
	target_link_libraries(native_headless_shutdown_order PRIVATE vcmi vcmiclientcommon vcmiservercommon TBB::tbb)
	vcmi_set_output_dir(native_headless_shutdown_order "")
endif()
