#ifndef MLP_MODEL_IO_IO_H_
#define MLP_MODEL_IO_IO_H_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../image.h"

namespace s21 {

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
using Dataset = std::vector<Image>;
using Weigths = std::vector<Matrix>;

Dataset ParseEmnist(const std::string& path);
void SaveWeights(const Weigths& weights, const std::string& path);
Weigths LoadWeights(const std::string& path);

}  // namespace s21

#endif  // MLP_MODEL_IO_IO_H_
