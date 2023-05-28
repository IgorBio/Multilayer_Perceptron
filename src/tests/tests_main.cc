#include <gtest/gtest.h>

#include "matrix_mlp_tests.cc"
#include "matrix_operations_tests.cc"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
