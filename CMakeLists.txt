cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(matmul_custom LANGUAGES ASC CXX)

# ---------------------------------------------------------------------------
# Host C++ standard. The device (kernel) side is compiled by bisheng with its
# own default dialect, so this only governs the plain C++ translation units.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Run mode of the produced binary: npu (real device), cpu (CPU debug), sim.
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "Run mode: npu, cpu, sim")

# Target SoC architecture. dav-2201 = Ascend 910B / Atlas A2.
set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture, e.g. dav-2201")

# Default to an optimised build; -DCMAKE_BUILD_TYPE=Debug flips the host side.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (Release/Debug/RelWithDebInfo/MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel)
endif()
message(STATUS "CMAKE_BUILD_TYPE='${CMAKE_BUILD_TYPE}'")

option(ENABLE_ASC_DEBUG "Build the AscendC kernel unoptimised with debug info" OFF)

add_executable(matmul_custom main.asc)

# Tiling generation (host) and the kernel both live in the single ASC TU; these
# are the same libraries the official matmul samples link against.
target_link_libraries(matmul_custom PRIVATE
    tiling_api
    register
    platform
    dl
    m
)

target_include_directories(matmul_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---------------------------------------------------------------------------
# Optimisation flags. main.asc is one bisheng invocation covering both the host
# wrapper and the device kernel, so CMAKE_BUILD_TYPE alone does not reach the
# .asc TU - it needs its own -O flag.
# ---------------------------------------------------------------------------
set(ASC_OPT_FLAGS)
if(ENABLE_ASC_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -Xaicore-start/-Xaicore-end scope -O0 -g to the device code only.
    list(APPEND ASC_OPT_FLAGS -Xaicore-start -O0 -Xaicore-end -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    list(APPEND ASC_OPT_FLAGS -Os)
else()
    list(APPEND ASC_OPT_FLAGS -O2)
endif()
message(STATUS "ASC optimisation flags: ${ASC_OPT_FLAGS}")

target_compile_options(matmul_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:${ASC_OPT_FLAGS}>
    "-gdwarf-4"
)
