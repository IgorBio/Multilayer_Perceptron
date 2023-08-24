cmake_minimum_required(VERSION 3.15)

project(Tests)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)

FetchContent_MakeAvailable(googletest)

target_compile_options(gtest PRIVATE "-w")
target_compile_options(gmock PRIVATE "-w") 

include_directories(
  ${PROJECT_SOURCE_DIR}/../model/utility
)

add_executable(${PROJECT_NAME}
  ${PROJECT_SOURCE_DIR}/../model/utility/matrix_operations.cc
  matrix_operations_tests.cc
)

add_executable(Emnist
  ${PROJECT_SOURCE_DIR}/../model/utility/io.cc
  parse_emnist_tests.cc
)

add_executable(Speed
  ${PROJECT_SOURCE_DIR}/../model/utility/io.cc
  ${PROJECT_SOURCE_DIR}/../model/utility/matrix_operations.cc
  speed_matrix_ops.cc
)

target_link_libraries(${PROJECT_NAME} PUBLIC gtest gtest_main)

target_compile_options(
    ${PROJECT_NAME}
    PUBLIC
    -Wall
    -Werror
    -Wextra
    -Wpedantic
    --coverage
    -O3
    -std=c++17
)

target_compile_options(Emnist PRIVATE -O3 -std=c++17)
target_compile_options(Speed PRIVATE -O3 -std=c++17)

target_link_options(${PROJECT_NAME} PRIVATE --coverage)
target_link_libraries(${PROJECT_NAME} PRIVATE -lgtest -lgtest_main)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

add_custom_target(coverage
  COMMAND lcov --directory . --capture --output-file coverage_report.info
  --exclude *_deps* --exclude *include/* --exclude *gtest*
  COMMAND genhtml coverage_report.info --output-directory gcov_report
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
