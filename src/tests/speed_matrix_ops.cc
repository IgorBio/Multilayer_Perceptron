#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "io.h"
#include "matrix_operations.h"

using namespace s21;

int main() {
  system("clear");
  Matrix m1 = Matrix(1000, Vector(1000));
  Matrix m2 = Matrix(1000, Vector(1000));
  RandomizeMatrix(m1);
  RandomizeMatrix(m2);
  double d = 0.1;

  std::cout << "\n"
            << GetColor(Color::kCyan)
            << Align("1000x1000 MATRIX OPERATIONS SPEED TEST")
            << GetColor(Color::kEnd) << "\n\n";

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) RandomizeMatrix(m1);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Randomize matrix: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = Addition(m1, m2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Addition matrices: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = Subtraction(m1, m2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Subtraction matrices: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = MultiplyNumber(m1, d);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Multiply number: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = MultiplyHadamard(m1, m2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Multiply Hadamard: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  Matrix res = Multiplication(m1, m2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Multiplication: " << std::to_string(elapsed.count())
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  res = MultiplyWinograd(m1, m2);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Multiply Winograd: " << std::to_string(elapsed.count())
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = Transpose(m1);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Transpose matrix: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i) Matrix res = Activate(m1, sigmoid);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Activate matrix: " << std::to_string(elapsed.count() / 100)
            << " sec\n";
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i)
    Matrix res = ActivateDerivative(m1, sigmoid_derivative);
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Activate Derivative matrix: "
            << std::to_string(elapsed.count() / 100) << " sec\n\n";
  std::cout << GetColor(Color::kCyan) << Align(" ") << GetColor(Color::kEnd)
            << "\n\n";
}
