# CMakeLists.txt
#
# Builds the standalone hybrid frontend modules
# (DormantTrackBuffer, SpatialDescriptorMatcher) and their unit tests.
# Has NO dependency on ORB-SLAM3 or OpenCV — these modules are pure
# data-structure / algorithm code per the design doc §9.2.

cmake_minimum_required(VERSION 3.14)
project(hybrid_frontend_modules CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Strict warnings — we want this to compile clean.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion
                        -Wno-sign-conversion)  # std::size_t vs int churn
endif()

# Library: the two modules built as a single static lib.
add_library(hybrid_frontend
    src/DormantTrackBuffer.cc
    src/SpatialDescriptorMatcher.cc
)
target_include_directories(hybrid_frontend PUBLIC include)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
enable_testing()
find_package(GTest REQUIRED)

add_executable(test_DormantTrackBuffer
    tests/test_DormantTrackBuffer.cc
)
target_link_libraries(test_DormantTrackBuffer
    PRIVATE hybrid_frontend GTest::gtest GTest::gtest_main
)
add_test(NAME DormantTrackBuffer COMMAND test_DormantTrackBuffer)

add_executable(test_SpatialDescriptorMatcher
    tests/test_SpatialDescriptorMatcher.cc
)
target_link_libraries(test_SpatialDescriptorMatcher
    PRIVATE hybrid_frontend GTest::gtest GTest::gtest_main
)
add_test(NAME SpatialDescriptorMatcher COMMAND test_SpatialDescriptorMatcher)
