# CMakeList.txt : CMake project for STREED, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.9)

project ("STreeD")
include(CTest)
set (CMAKE_CXX_STANDARD 17)

# Set release build.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
	message(STATUS "Build type not specified: Use Release by default")
endif ()
if (CMAKE_BUILD_TYPE STREQUAL Release)
	message(STATUS "Compiling in release mode")
else()
	message(STATUS "Compiling in debug mode")	
	add_compile_definitions(DEBUG=1)
endif()

# Headers
set(PROJECT_SOURCE_DIR "src")
set(PROJECT_INCLUDE_DIR "include")

set(STREED_HEADER_FILES
	${PROJECT_INCLUDE_DIR}/base.h

	${PROJECT_INCLUDE_DIR}/utils/debug.h
	${PROJECT_INCLUDE_DIR}/utils/parameter_handler.h
	${PROJECT_INCLUDE_DIR}/utils/stopwatch.h
	${PROJECT_INCLUDE_DIR}/utils/file_reader.h
	${PROJECT_INCLUDE_DIR}/utils/key_value_heap.h
	${PROJECT_INCLUDE_DIR}/utils/dynamic_bitset.h
	
	${PROJECT_INCLUDE_DIR}/model/data.h
	${PROJECT_INCLUDE_DIR}/model/feature_vector.h
	${PROJECT_INCLUDE_DIR}/model/branch.h
	${PROJECT_INCLUDE_DIR}/model/node.h
	${PROJECT_INCLUDE_DIR}/model/container.h
	

	${PROJECT_INCLUDE_DIR}/solver/result.h
	${PROJECT_INCLUDE_DIR}/solver/feature_selector.h
	${PROJECT_INCLUDE_DIR}/solver/difference_computer.h
	${PROJECT_INCLUDE_DIR}/solver/similarity_lowerbound.h
	${PROJECT_INCLUDE_DIR}/solver/cost_combiner.h
	${PROJECT_INCLUDE_DIR}/solver/cost_storage.h
	${PROJECT_INCLUDE_DIR}/solver/counter.h
	${PROJECT_INCLUDE_DIR}/solver/tree.h
	${PROJECT_INCLUDE_DIR}/solver/cache.h
	${PROJECT_INCLUDE_DIR}/solver/dataset_cache.h
	${PROJECT_INCLUDE_DIR}/solver/branch_cache.h
	${PROJECT_INCLUDE_DIR}/solver/terminal_solver.h
	${PROJECT_INCLUDE_DIR}/solver/statistics.h
	${PROJECT_INCLUDE_DIR}/solver/solver.h
	${PROJECT_INCLUDE_DIR}/solver/optimization_utils.h
	${PROJECT_INCLUDE_DIR}/solver/data_splitter.h
	${PROJECT_INCLUDE_DIR}/solver/tuning.h

	${PROJECT_INCLUDE_DIR}/tasks/tasks.h
	${PROJECT_INCLUDE_DIR}/tasks/optimization_task.h
	${PROJECT_INCLUDE_DIR}/tasks/cost_sensitive.h
	${PROJECT_INCLUDE_DIR}/tasks/f1score.h
	${PROJECT_INCLUDE_DIR}/tasks/group_fairness.h
	${PROJECT_INCLUDE_DIR}/tasks/eq_opp.h
	${PROJECT_INCLUDE_DIR}/tasks/prescriptive_policy.h

	${PROJECT_INCLUDE_DIR}/tasks/accuracy/accuracy.h
	${PROJECT_INCLUDE_DIR}/tasks/accuracy/cost_complex_accuracy.h

)

set(STREED_SRC_FILES

	${PROJECT_SOURCE_DIR}/utils/parameter_handler.cpp
	${PROJECT_SOURCE_DIR}/utils/file_reader.cpp
	${PROJECT_SOURCE_DIR}/utils/key_value_heap.cpp

	${PROJECT_SOURCE_DIR}/model/feature_vector.cpp
	${PROJECT_SOURCE_DIR}/model/data.cpp
	${PROJECT_SOURCE_DIR}/model/branch.cpp

	${PROJECT_SOURCE_DIR}/solver/feature_selector_gini.cpp
	${PROJECT_SOURCE_DIR}/solver/define_parameters.cpp
	${PROJECT_SOURCE_DIR}/solver/solver.cpp
	${PROJECT_SOURCE_DIR}/solver/cache.cpp
	${PROJECT_SOURCE_DIR}/solver/branch_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/dataset_cache.cpp
	${PROJECT_SOURCE_DIR}/solver/cost_combiner.cpp
	${PROJECT_SOURCE_DIR}/solver/terminal_solver.cpp
	${PROJECT_SOURCE_DIR}/solver/cost_storage.cpp
	${PROJECT_SOURCE_DIR}/solver/counter.cpp
	${PROJECT_SOURCE_DIR}/solver/result.cpp
	${PROJECT_SOURCE_DIR}/solver/similarity_lowerbound.cpp
	${PROJECT_SOURCE_DIR}/solver/difference_computer.cpp
	${PROJECT_SOURCE_DIR}/solver/data_splitter.cpp

	${PROJECT_SOURCE_DIR}/tasks/optimization_task.cpp
	${PROJECT_SOURCE_DIR}/tasks/cost_sensitive.cpp
	${PROJECT_SOURCE_DIR}/tasks/f1score.cpp
	${PROJECT_SOURCE_DIR}/tasks/group_fairness.cpp
	${PROJECT_SOURCE_DIR}/tasks/eq_opp.cpp
	${PROJECT_SOURCE_DIR}/tasks/prescriptive_policy.cpp

	${PROJECT_SOURCE_DIR}/tasks/accuracy/accuracy.cpp
	${PROJECT_SOURCE_DIR}/tasks/accuracy/cost_complex_accuracy.cpp


	${STREED_HEADER_FILES}
)

if (CMAKE_BUILD_TYPE STREQUAL Release)
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Add source to this project's executable.
add_library(STREEDLib ${STREED_SRC_FILES})
add_executable (STREED ${PROJECT_SOURCE_DIR}/main.cpp)

target_include_directories(STREEDLib PUBLIC ${PROJECT_INCLUDE_DIR})
target_link_libraries (STREED STREEDLib)

if (CMAKE_BUILD_TYPE STREQUAL Release)
else()
	if (MSVC)
		message(STATUS "Enabling profiler")	
		#set_target_properties(STREEDLib STREED PROPERTIES LINK_FLAGS "/PROFILE")
		SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
	else ()
		message(STATUS "No profiler")
	endif()
endif()

# Tests

enable_testing()
set(PROJECT_TEST_DIR "test")
add_executable(CorrectSolutionTest ${PROJECT_TEST_DIR}/correct_solution_test.cpp)
target_link_libraries (CorrectSolutionTest STREEDLib)
add_test(NAME "CorrectSolutionTest" COMMAND CorrectSolutionTest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
