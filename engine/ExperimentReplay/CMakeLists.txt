# Replays an experiment's recorded server inputs and verifies its digests.
add_executable(vcmireplay StdInc.cpp StdInc.h ReplayMain.cpp)
target_link_libraries(vcmireplay PRIVATE vcmi vcmiservercommon)
target_include_directories(vcmireplay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
vcmi_set_output_dir(vcmireplay "")
enable_pch(vcmireplay)
