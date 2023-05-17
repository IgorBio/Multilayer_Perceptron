#include <gtest/gtest.h>

#include <cmath>

#include "../model/utility/matrix_operations.h"

constexpr double eps = 1e-7;

bool IsEqualMatrices(const s21::Matrix& m1, const s21::Matrix& m2);
bool IsEqualDouble(const double d1, const double d2);
void PrintMatrix(s21::Matrix m);

bool IsEqualMatrices(const s21::Matrix& m1, const s21::Matrix& m2) {
  if (m1.size() != m2.size() or m1[0].size() != m2[0].size()) return false;
  for (std::size_t i{0u}; i < m1.size(); ++i) {
    for (std::size_t j{0u}; j < m1[0].size(); ++j) {
      if (!IsEqualDouble(m1[i][j], m2[i][j])) return false;
    }
  }
  return true;
}

bool IsEqualDouble(const double d1, const double d2) {
  return std::fabs(d1 - d2) < eps;
}

void PrintMatrix(s21::Matrix m) {
  for (std::size_t i = 0; i < m.size(); i++) {
    for (std::size_t j = 0; j < m[i].size(); j++) {
      std::cout << m[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

TEST(Randomize, CorrectCase) {
  s21::Matrix m = s21::Matrix(1000, s21::Vector(1000));
  EXPECT_NO_THROW(s21::Randomize(m));
}

TEST(Subtraction, CorrectCase) {
  s21::Matrix m1 = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  s21::Matrix m2 = {{-1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  s21::Matrix expected = {{2, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
  s21::Matrix res = s21::Subtraction(m1, m2);
  EXPECT_TRUE(IsEqualMatrices(res, expected));
}

TEST(Transpose, CorrectCase) {
  s21::Matrix m = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  s21::Matrix expected = {
      {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 6}, {5, 6, 7}};
  s21::Matrix res = s21::Transpose(m);
  EXPECT_TRUE(IsEqualMatrices(res, expected));
}

TEST(MultiplyNumber, CorrectCase) {
  s21::Matrix m = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  double d = 5;
  s21::Matrix expected = {
      {5, 10, 15, 20, 25}, {10, 15, 20, 25, 30}, {15, 20, 25, 30, 35}};
  s21::Matrix res = s21::MultiplyNumber(m, d);
  EXPECT_TRUE(IsEqualMatrices(res, expected));
}

TEST(MultiplyHadamard, CorrectCase) {
  s21::Matrix m1 = {{1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  s21::Matrix m2 = {{-1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}, {3, 4, 5, 6, 7}};
  s21::Matrix expected = {
      {-1, 4, 9, 16, 25}, {4, 9, 16, 25, 36}, {9, 16, 25, 36, 49}};
  s21::Matrix res = s21::MultiplyHadamard(m1, m2);
  EXPECT_TRUE(IsEqualMatrices(res, expected));
}
