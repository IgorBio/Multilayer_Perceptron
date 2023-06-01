#include "io.h"

namespace s21 {

Dataset ParseEmnist(const std::string& path) {
  Dataset dataset;
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::string line, pixel_str;
  while (std::getline(file, line)) {
    Image image;
    std::istringstream iss(line);
    std::string token;
    std::getline(iss, token, ',');
    image.SetLabel(std::stoi(token));
    for (std::size_t i{0u}; i < Image::kPixels; ++i) {
      std::getline(iss, token, ',');
      image.AddPixel(static_cast<double>(std::stoi(token)) / Image::kMaxPixel);
    }
    dataset.push_back(std::move(image));
  }
  return std::move(dataset);
}

void SaveWeights(const Tensor& weights, const std::string& path) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  // write the number of layers
  std::size_t num_layers = weights.size();
  file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

  // write each layer's weights
  for (const Matrix& layer_weights : weights) {
    // write the dimensions of the weight matrix
    std::size_t rows = layer_weights.size();
    std::size_t cols = layer_weights[0].size();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // write the weight matrix
    for (const Vector& row : layer_weights) {
      file.write(reinterpret_cast<const char*>(row.data()),
                 sizeof(double) * cols);
    }
  }
}

Tensor LoadWeights(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }

  // read the number of layers
  std::size_t num_layers;
  file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

  // read each layer's weights
  Tensor weights(num_layers);
  for (std::size_t i{0}; i < num_layers; ++i) {
    // read the dimensions of the weight matrix
    std::size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // read the weight matrix
    Matrix layer_weights(rows, Vector(cols));
    for (Vector& row : layer_weights) {
      file.read(reinterpret_cast<char*>(row.data()), sizeof(double) * cols);
    }
    weights[i] = std::move(layer_weights);
  }

  return weights;
}

}  // namespace s21
