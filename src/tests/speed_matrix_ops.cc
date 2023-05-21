#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "../model/utility/matrix_operations.h"

constexpr size_t kWidth = 60u;

enum class Color { kRed, kGreen, kBlue, kYellow, kGrey, kCyan, kMagenta, kEnd };
std::string GetColor(Color color) {
  switch (color) {
    case Color::kRed:
      return "\u001b[41;1m";
    case Color::kGreen:
      return "\u001b[42;1m";
    case Color::kYellow:
      return "\u001b[43;1m";
    case Color::kBlue:
      return "\u001b[44;1m";
    case Color::kMagenta:
      return "\u001b[45;1m";
    case Color::kCyan:
      return "\u001b[46;1m";
    case Color::kGrey:
      return "\u001b[47;1m";
    case Color::kEnd:
      return "\u001b[0m";
    default:
      return "";
  }
}

std::string Align(const std::string &str) {
  std::string aligned;
  std::string addition((kWidth - str.size()) / 2, ' ');
  aligned.append(addition);
  aligned.append(str);
  aligned.append(addition);
  while (aligned.size() < kWidth) aligned.append(" ");
  return aligned;
}

int main() {
  s21::Matrix m1 = s21::Matrix(1000, s21::Vector(1000));
  s21::Matrix m2 = s21::Matrix(1000, s21::Vector(1000));
  s21::RandomizeMatrix(m1);
  s21::RandomizeMatrix(m2);
  double d = 0.1;

  std::cout << GetColor(Color::kCyan)
            << Align("1000x1000 MATRIX OPERATIONS SPEED TEST")
            << GetColor(Color::kEnd) << "\n\n";

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::RandomizeMatrix(m1);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "   RandomizeMatrix matrix: "
            << std::to_string(elapsed.count() / 100) << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::Matrix res = s21::Addition(m1, m2);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Addition matrices: " << std::to_string(elapsed.count() / 100)
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::Matrix res = s21::Subtraction(m1, m2);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Subtraction matrices: "
            << std::to_string(elapsed.count() / 100) << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::Matrix res = s21::MultiplyNumber(m1, d);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Multiply number: " << std::to_string(elapsed.count() / 100)
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::Matrix res = s21::MultiplyHadamard(m1, m2);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Multiply Hadamard: " << std::to_string(elapsed.count() / 100)
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  s21::Matrix res = s21::MultiplyWinograd(m1, m2);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Multiply Winograd: " << std::to_string(elapsed.count())
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i) s21::Matrix res = s21::Transpose(m1);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Transpose matrix: " << std::to_string(elapsed.count() / 100)
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i)
    s21::Matrix res = s21::Activate(m1, s21::ActivationFunction::kSigmoid);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Activate matrix: " << std::to_string(elapsed.count() / 100)
            << " sec"
            << "\n";
  start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; ++i)
    s21::Matrix res =
        s21::DeriveActivate(m1, s21::ActivationFunction::kSigmoid);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "   Derive Activate matrix: "
            << std::to_string(elapsed.count() / 100) << " sec"
            << "\n\n";
  std::cout << GetColor(Color::kCyan) << Align(" ") << GetColor(Color::kEnd)
            << "\n";
}
