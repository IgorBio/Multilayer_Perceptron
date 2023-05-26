#ifndef MLP_MODEL_IO_EMNIST_PARSER_H
#define MLP_MODEL_IO_EMNIST_PARSER_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "image.h"

namespace s21 {
class EmnistParser {
 public:
  using Dataset = std::vector<Image>;

  Dataset ParseEmnist(const std::string& path);
};
}  // namespace s21

#endif  // MLP_MODEL_IO_EMNIST_PARSER_H
