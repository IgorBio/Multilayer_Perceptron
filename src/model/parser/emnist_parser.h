#ifndef SRC_MODEL_PARSER_EMNIST_PARSER_H
#define SRC_MODEL_PARSER_EMNIST_PARSER_H

#include <fstream>
#include <string>
#include <vector>

#include "image.h"

namespace s21 {
class EmnistParser {
 public:
  using Dataset = std::vector<Image>;

  Dataset ParseEmnist(const std::string& path);

 private:
  int ParseLabel(const std::string& line, std::size_t in, std::size_t* out);
  double ParsePixel(const std::string& line, std::size_t in, std::size_t* out);
};
}  // namespace s21

#endif  // SRC_MODEL_PARSER_EMNIST_PARSER_H
