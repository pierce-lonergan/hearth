cmake_minimum_required(VERSION 3.20)
project(hearth VERSION 0.1.0 LANGUAGES C)

# Hearth: zero-dependency C11 engine. Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release
#   ctest --test-dir build -C Release
# On Windows prefer scripts/build.ps1 (keeps build output out of synced folders).

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
option(HEARTH_TESTS "Build C unit tests" ON)
option(HEARTH_ASAN "Build with AddressSanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/engine/src)
set(CORE_SOURCES
  ${SRC}/pool.c
  ${SRC}/quant.c
  ${SRC}/modelfile.c
  ${SRC}/store.c
  ${SRC}/model.c
  ${SRC}/api.c)
if(WIN32)
  list(APPEND CORE_SOURCES ${SRC}/platform_win.c)
else()
  list(APPEND CORE_SOURCES ${SRC}/platform_posix.c)
endif()

# ISA-specific kernels get per-file flags; dispatch happens at runtime.
set(X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|amd64|x64")
  set(X86 ON)
endif()
if(X86)
  list(APPEND CORE_SOURCES ${SRC}/quant_avx2.c ${SRC}/quant_avx512.c)
  if(MSVC)
    set_source_files_properties(${SRC}/quant_avx2.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(${SRC}/quant_avx512.c PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(${SRC}/quant_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(${SRC}/quant_avx512.c PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni;-mavx2;-mfma;-mf16c")
  endif()
endif()

add_library(hearth_core OBJECT ${CORE_SOURCES})
target_include_directories(hearth_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine/include ${SRC})
set_target_properties(hearth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(hearth_core PRIVATE HEARTH_BUILD_DLL)
if(MSVC)
  # Numerics contract (docs/NUMERICS.md): precise FP, no contraction.
  target_compile_options(hearth_core PUBLIC /experimental:c11atomics /fp:precise /utf-8 /W3)
  target_compile_definitions(hearth_core PUBLIC _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(hearth_core PUBLIC -ffp-contract=off -Wall -Wextra -Wno-unused-parameter)
endif()
if(HEARTH_ASAN)
  if(MSVC)
    target_compile_options(hearth_core PUBLIC /fsanitize=address)
  else()
    target_compile_options(hearth_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(hearth_core PUBLIC -fsanitize=address,undefined)
  endif()
endif()
find_package(Threads REQUIRED)
target_link_libraries(hearth_core PUBLIC Threads::Threads)
if(NOT WIN32)
  target_link_libraries(hearth_core PUBLIC m)
endif()

add_library(hearth SHARED $<TARGET_OBJECTS:hearth_core>)
target_link_libraries(hearth PRIVATE hearth_core)
set_target_properties(hearth PROPERTIES C_VISIBILITY_PRESET hidden)

add_executable(hearth-bench engine/tools/hearth_bench.c)
target_link_libraries(hearth-bench PRIVATE hearth_core)

if(HEARTH_TESTS)
  enable_testing()
  file(GLOB TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/engine/tests/test_*.c)
  foreach(t ${TEST_SOURCES})
    get_filename_component(name ${t} NAME_WE)
    add_executable(${name} ${t})
    target_link_libraries(${name} PRIVATE hearth_core)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
endif()
