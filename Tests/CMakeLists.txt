# ORB_SLAM3/Tests/CMakeLists.txt
#
# Unit tests for the hybrid frontend modules.
#
# ARCHITECTURE: this file builds TESTS ONLY. It defines no library.
#
# The modules themselves live in their canonical §9.1 location —
# ../include/DormantTrackBuffer.h, ../src/DormantTrackBuffer.cc, and the
# same for SpatialDescriptorMatcher — where the root CMakeLists.txt's
# GLOB_RECURSE over src/*.cc already picks them up and compiles them into
# libORB_SLAM3.so. There is exactly one copy of each source file and one
# place that builds it for the library.
#
# The test executables compile those same .cc files directly rather than
# linking libORB_SLAM3.so. That is deliberate: the modules have no
# ORB-SLAM3, OpenCV, Pangolin or g2o dependency (§9.2 step 1 requires
# them to be testable in isolation), so linking the full library would
# drag in every heavy dependency to test two header-only-ish data
# structures, and would couple test success to the library building.
#
# This directory is NOT added by the root CMakeLists.txt. Configure it
# on its own:
#
#     cd ORB_SLAM3/Tests && mkdir -p build && cd build
#     cmake .. -DCMAKE_BUILD_TYPE=Debug
#     make -j && ctest --output-on-failure
#
# Debug matters: the §4.1 / §7.4.2 invariants are enforced with assert(),
# so the EXPECT_DEATH tests are compiled out under NDEBUG.

cmake_minimum_required(VERSION 3.14)
project(hybrid_frontend_tests CXX)

# C++17 for std::optional in SpatialDescriptorMatcher.h. NOTE: the root
# ORB-SLAM3 CMakeLists.txt currently forces -std=c++14; it must be
# raised to 17 before these modules will compile into libORB_SLAM3.so.
# See INTEGRATION_NOTES.md.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion
                        -Wno-sign-conversion)  # std::size_t vs int churn
endif()

set(HYBRID_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(HYBRID_MODULE_SOURCES
    ${HYBRID_MODULE_DIR}/src/DormantTrackBuffer.cc
    ${HYBRID_MODULE_DIR}/src/SpatialDescriptorMatcher.cc
)
foreach(f ${HYBRID_MODULE_SOURCES})
    if(NOT EXISTS ${f})
        message(FATAL_ERROR
            "Missing ${f}\n"
            "The hybrid modules belong in ORB_SLAM3/src and ORB_SLAM3/include, "
            "where the root CMakeLists.txt GLOB picks them up.")
    endif()
endforeach()

# ---------------------------------------------------------------------
# GoogleTest
# ---------------------------------------------------------------------
# -DHYBRID_FETCH_GTEST=ON builds GoogleTest from source with this
# project's exact flags. Immune to whatever is installed system-wide or
# in a conda environment. Use it if the RTTI check below fails.
option(HYBRID_FETCH_GTEST "Download and build GoogleTest from source" OFF)

if(HYBRID_FETCH_GTEST)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(HYBRID_GTEST_LIBS GTest::gtest GTest::gtest_main)
    message(STATUS "GoogleTest: built from source (FetchContent v1.14.0)")
else()
    # A conda environment puts $CONDA_PREFIX ahead of /usr in CMake's
    # search path, and conda's googletest is frequently built without
    # RTTI. That produces a wall of
    #   undefined reference to `typeinfo for testing::Test'
    # at link time, because the library never emits that symbol while
    # every TEST() class in this directory references it. Pass
    # -DHYBRID_ALLOW_CONDA_GTEST=ON to skip this exclusion.
    option(HYBRID_ALLOW_CONDA_GTEST "Permit a GTest from \$CONDA_PREFIX" OFF)
    if(NOT HYBRID_ALLOW_CONDA_GTEST AND DEFINED ENV{CONDA_PREFIX})
        message(STATUS "Conda detected at $ENV{CONDA_PREFIX} — excluding it from GTest search")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
        list(APPEND CMAKE_IGNORE_PATH "$ENV{CONDA_PREFIX}" "$ENV{CONDA_PREFIX}/lib"
                                      "$ENV{CONDA_PREFIX}/include")
    endif()

    find_package(GTest REQUIRED)
    set(HYBRID_GTEST_LIBS GTest::gtest GTest::gtest_main)

    # Report exactly which library was selected, so this is never a
    # mystery again. Imported targets from GTestConfig.cmake carry their
    # path in IMPORTED_LOCATION_<CONFIG>, and the config names vary by
    # distribution, so ask the target which ones it has.
    set(_gtest_loc "")
    get_target_property(_gtest_cfgs GTest::gtest IMPORTED_CONFIGURATIONS)
    if(_gtest_cfgs)
        foreach(_cfg ${_gtest_cfgs})
            get_target_property(_try GTest::gtest IMPORTED_LOCATION_${_cfg})
            if(_try)
                set(_gtest_loc ${_try})
                break()
            endif()
        endforeach()
    endif()
    if(NOT _gtest_loc)
        get_target_property(_try GTest::gtest IMPORTED_LOCATION)
        if(_try)
            set(_gtest_loc ${_try})
        endif()
    endif()

    if(_gtest_loc)
        message(STATUS "GoogleTest library: ${_gtest_loc}")
    else()
        message(WARNING "Could not determine the GoogleTest library path; "
                        "skipping the RTTI check.")
    endif()

    # Configure-time RTTI check. Fail here with an actionable message
    # rather than after a full compile with dozens of linker errors.
    find_program(NM_EXECUTABLE nm)
    if(NOT NM_EXECUTABLE)
        message(WARNING "'nm' not found; skipping the GTest RTTI check.")
    elseif(_gtest_loc AND EXISTS ${_gtest_loc})
        execute_process(
            COMMAND ${NM_EXECUTABLE} -C --defined-only ${_gtest_loc}
            OUTPUT_VARIABLE _nm_out ERROR_QUIET)
        string(FIND "${_nm_out}" "typeinfo for testing::Test" _ti_pos)
        if(_ti_pos EQUAL -1)
            message(FATAL_ERROR
                "\n"
                "The GoogleTest at:\n"
                "    ${_gtest_loc}\n"
                "does not export 'typeinfo for testing::Test'. It was built "
                "without RTTI (-fno-rtti / GTEST_HAS_RTTI=0), so linking will "
                "fail with:\n"
                "    undefined reference to `typeinfo for testing::Test'\n"
                "\n"
                "Pick one:\n"
                "  1. Build GoogleTest from source (most reliable):\n"
                "       cmake .. -DHYBRID_FETCH_GTEST=ON\n"
                "  2. Leave the conda environment and reconfigure from clean:\n"
                "       conda deactivate && rm -rf build && mkdir build && cd build && cmake ..\n"
                "  3. Point at a known-good install:\n"
                "       cmake .. -DGTEST_ROOT=/usr\n"
                "\n"
                "Reconfigure from a CLEAN build directory — CMakeCache.txt "
                "caches the bad path.\n")
        endif()
        message(STATUS "GoogleTest RTTI check: OK")
    endif()
endif()

# ---------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------
enable_testing()

add_executable(test_DormantTrackBuffer
    test_DormantTrackBuffer.cc
    ${HYBRID_MODULE_DIR}/src/DormantTrackBuffer.cc)
target_include_directories(test_DormantTrackBuffer PRIVATE ${HYBRID_MODULE_DIR}/include)
target_link_libraries(test_DormantTrackBuffer PRIVATE ${HYBRID_GTEST_LIBS})
add_test(NAME DormantTrackBuffer COMMAND test_DormantTrackBuffer)

add_executable(test_SpatialDescriptorMatcher
    test_SpatialDescriptorMatcher.cc
    ${HYBRID_MODULE_DIR}/src/SpatialDescriptorMatcher.cc)
target_include_directories(test_SpatialDescriptorMatcher PRIVATE ${HYBRID_MODULE_DIR}/include)
target_link_libraries(test_SpatialDescriptorMatcher PRIVATE ${HYBRID_GTEST_LIBS})
add_test(NAME SpatialDescriptorMatcher COMMAND test_SpatialDescriptorMatcher)

# ---- HybridFrontend tests (need OpenCV; other targets stay OpenCV-free)
# Preferred: the OpenCVConfig.cmake that ORB-SLAM3's own build already
# requires. Fallback: pkg-config (covers minimal environments that have
# the module -dev packages but not the CMake config).
find_package(OpenCV QUIET COMPONENTS core imgproc video features2d calib3d)
if(NOT OpenCV_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENCV_PC QUIET opencv4)
        if(OPENCV_PC_FOUND)
            set(OpenCV_FOUND TRUE)
            set(OpenCV_VERSION ${OPENCV_PC_VERSION})
            set(OpenCV_INCLUDE_DIRS ${OPENCV_PC_INCLUDE_DIRS})
            set(OpenCV_LIBS opencv_core opencv_imgproc opencv_video
                            opencv_features2d opencv_calib3d)
            link_directories(${OPENCV_PC_LIBRARY_DIRS})
        endif()
    endif()
endif()
if(OpenCV_FOUND)
    message(STATUS "OpenCV ${OpenCV_VERSION} found - building HybridFrontend tests")
    add_executable(test_HybridFrontend
        test_HybridFrontend.cc
        ${HYBRID_MODULE_DIR}/src/HybridFrontend.cc
        ${HYBRID_MODULE_DIR}/src/DormantTrackBuffer.cc
        ${HYBRID_MODULE_DIR}/src/SpatialDescriptorMatcher.cc)
    target_include_directories(test_HybridFrontend PRIVATE
        ${HYBRID_MODULE_DIR}/include)
    target_include_directories(test_HybridFrontend SYSTEM PRIVATE
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_HybridFrontend PRIVATE
        ${HYBRID_GTEST_LIBS} ${OpenCV_LIBS})
    add_test(NAME HybridFrontend COMMAND test_HybridFrontend)
else()
    message(STATUS "OpenCV not found - skipping HybridFrontend tests")
endif()

# Cross-language parity harness (no GTest). Diff its stdout against
# `python3 parity_dump.py --module-dir <dir with dormant_buffer.py>`;
# the two must be byte-identical.
add_executable(parity_dump
    parity_dump.cc
    ${HYBRID_MODULE_DIR}/src/DormantTrackBuffer.cc
    ${HYBRID_MODULE_DIR}/src/SpatialDescriptorMatcher.cc)
target_include_directories(parity_dump PRIVATE ${HYBRID_MODULE_DIR}/include)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS
        "Build type is '${CMAKE_BUILD_TYPE}'. The §4.1/§7.4.2 invariant "
        "death-tests are compiled out under NDEBUG; use "
        "-DCMAKE_BUILD_TYPE=Debug to run them.")
endif()