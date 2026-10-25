# Opt-in live host; not linked into the ordinary client or Teacher runtime.
option(ENABLE_EMBEDDED_ENGINE "Build the in-process Python engine module (heroes_embedded_engine)" OFF)
if(ENABLE_EMBEDDED_ENGINE)
	# A Python extension is a shared object: every static archive inside it
	# must be position independent. Executables built in the same tree change
	# only their code generation, never game rules.
	set_target_properties(vcmiclientcommon vcmiservercommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_library(native_engine_session STATIC EngineSession.cpp EngineSession.h Opponent.cpp Opponent.h RuntimeEnvironment.cpp RuntimeEnvironment.h SessionHealth.h)
target_link_libraries(native_engine_session PUBLIC vcmi vcmiclientcommon vcmiservercommon)
target_include_directories(native_engine_session PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(ENABLE_EMBEDDED_ENGINE)
	set_target_properties(native_engine_session PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
add_executable(vcmisession SessionMain.cpp ResourceUsage.cpp ResourceUsage.h)
target_link_libraries(vcmisession PRIVATE native_engine_session)
add_dependencies(vcmisession ExplorerRL Nullkiller BattleAI)
vcmi_set_output_dir(vcmisession "")
enable_pch(native_engine_session)
enable_pch(vcmisession)
# Dedicated asset-free ownership admission fixture, built explicitly by tests.
add_executable(native_session_contract EXCLUDE_FROM_ALL tests/HostContract.cpp)
target_link_libraries(native_session_contract PRIVATE native_engine_session)
vcmi_set_output_dir(native_session_contract "")
# Dedicated, explicitly labelled real-engine lifecycle workload, never training.
add_executable(native_session_benchmark SessionBenchmark.cpp ResourceUsage.cpp)
target_link_libraries(native_session_benchmark PRIVATE native_engine_session)
add_dependencies(native_session_benchmark ExplorerRL Nullkiller BattleAI)
vcmi_set_output_dir(native_session_benchmark "")
enable_pch(native_session_benchmark)

if(ENABLE_EMBEDDED_ENGINE)
	# Configure with -DPython3_EXECUTABLE=<the learner virtualenv python>; the
	# module ABI must match the interpreter that imports it. CPython C API only.
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
	Python3_add_library(heroes_embedded_engine MODULE WITH_SOABI PythonModule.cpp ResourceUsage.cpp ResourceUsage.h)
	target_link_libraries(heroes_embedded_engine PRIVATE native_engine_session)
	add_dependencies(heroes_embedded_engine ExplorerRL Nullkiller BattleAI)
	# Kept outside bin/: the replay/runtime identity inventory there stays the
	# executable runtime. The module finds libvcmi beside the executables.
	set_target_properties(heroes_embedded_engine PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/embedded"
		BUILD_RPATH "${CMAKE_BINARY_DIR}/bin"
		INSTALL_RPATH "${CMAKE_BINARY_DIR}/bin")
endif()
