cmake_minimum_required(VERSION 3.22)

project(schedlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)

find_package(CUDAToolkit REQUIRED)

set(SCHEDLAB_TENSORRT_HINTS
  "$ENV{HOME}/.katago/tensorrt"
  "$ENV{HOME}/.katago/tensorrt-cu13-1015"
)

find_path(TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS ${TENSORRT_ROOT_DIR} ${SCHEDLAB_TENSORRT_HINTS}
  PATH_SUFFIXES include
  REQUIRED
)

find_path(TENSORRT_ONNXPARSER_INCLUDE_DIR
  NvOnnxParser.h
  HINTS ${TENSORRT_ROOT_DIR} ${SCHEDLAB_TENSORRT_HINTS}
  PATH_SUFFIXES include
  REQUIRED
)

find_library(TENSORRT_LIBRARY
  nvinfer
  HINTS ${TENSORRT_ROOT_DIR} ${SCHEDLAB_TENSORRT_HINTS}
  PATH_SUFFIXES lib
  REQUIRED
)

find_library(TENSORRT_ONNXPARSER_LIBRARY
  nvonnxparser
  HINTS ${TENSORRT_ROOT_DIR} ${SCHEDLAB_TENSORRT_HINTS}
  PATH_SUFFIXES lib
  REQUIRED
)

add_library(schedlab_core
  src/config.cpp
  src/infer_backend.cpp
  src/scheduler.cpp
  src/mock_phase_runner.cpp
  src/search.cpp
  src/dispatcher.cpp
  src/trt_backend.cpp
)

target_include_directories(schedlab_core
  PUBLIC
    include
    ../../third_party/stdexec/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${TENSORRT_ONNXPARSER_INCLUDE_DIR}
)

target_compile_options(schedlab_core
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wno-missing-field-initializers
)

target_link_libraries(schedlab_core
  PUBLIC
    pthread
    dl
    CUDA::cudart
    ${TENSORRT_LIBRARY}
    ${TENSORRT_ONNXPARSER_LIBRARY}
)

add_executable(schedlab_run
  src/main.cpp
)

target_link_libraries(schedlab_run PRIVATE schedlab_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(schedlab_sync_primitives_test
    tests/sync_primitives_test.cpp
  )

  target_include_directories(schedlab_sync_primitives_test
    PRIVATE
      include
      ../../third_party/stdexec/include
  )

  target_compile_options(schedlab_sync_primitives_test
    PRIVATE
      -Wall
      -Wextra
      -Wpedantic
      -Wconversion
      -Wno-missing-field-initializers
  )

  target_link_libraries(schedlab_sync_primitives_test PRIVATE pthread)

  add_test(
    NAME schedlab_one_shot_event_set_before_wait
    COMMAND schedlab_sync_primitives_test one_shot_set_before_wait
  )
  add_test(
    NAME schedlab_one_shot_event_reset_reuse
    COMMAND schedlab_sync_primitives_test one_shot_reset_reuse
  )
  add_test(
    NAME schedlab_one_shot_event_duplicate_waiter
    COMMAND schedlab_sync_primitives_test one_shot_duplicate_waiter_terminates
  )
  add_test(
    NAME schedlab_one_shot_event_reset_with_waiter
    COMMAND schedlab_sync_primitives_test one_shot_reset_with_waiter_terminates
  )
  add_test(
    NAME schedlab_one_shot_event_old_sender_after_reset
    COMMAND schedlab_sync_primitives_test one_shot_old_sender_after_reset_terminates
  )
  add_test(
    NAME schedlab_pause_gate_initially_open
    COMMAND schedlab_sync_primitives_test pause_gate_initially_open
  )
  add_test(
    NAME schedlab_pause_gate_sender_lifetime
    COMMAND schedlab_sync_primitives_test pause_gate_sender_lifetime
  )
  add_test(
    NAME schedlab_pause_gate_broadcast
    COMMAND schedlab_sync_primitives_test pause_gate_broadcast
  )
  add_test(
    NAME schedlab_pause_gate_destroyed_owner_waiter
    COMMAND schedlab_sync_primitives_test pause_gate_destroyed_owner_releases_waiter
  )
  add_test(
    NAME schedlab_pause_gate_cancelling_waiter
    COMMAND schedlab_sync_primitives_test pause_gate_cancelling_waiter_terminates
  )
  add_test(
    NAME schedlab_pause_gate_force_open
    COMMAND schedlab_sync_primitives_test pause_gate_force_open_sticky
  )
  add_test(NAME schedlab_run_smoke COMMAND ${CMAKE_SOURCE_DIR}/run.sh --run-ms 200)
endif()
