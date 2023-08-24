#ifndef MLP_MODEL_UTILITY_IO_H_
#define MLP_MODEL_UTILITY_IO_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../image.h"

namespace s21 {

using Dataset = std::vector<Image>;
constexpr std::size_t kStringWidth = 40u;
enum class Color { kRed, kGreen, kBlue, kYellow, kGrey, kCyan, kMagenta, kEnd };

Dataset ParseEmnist(const std::string& path);
std::string GetColor(Color color);
std::string Align(const std::string& str);

}  // namespace s21

#endif  // MLP_MODEL_UTILITY_IO_H_
