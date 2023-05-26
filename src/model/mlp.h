#ifndef MLP_MODEL_MLP_H_
#define MLP_MODEL_MLP_H_

#include "config.h"
#include "graph_mlp/graph_mlp.h"
#include "io/emnist_parser.h"
#include "matrix_mlp/matrix_mlp.h"
#include "mlp_abstract.h"

namespace s21 {

class MLP {
 public:
  explicit MLP(ModelType type, Topology topology);

  void Train();
  void Test();
  Vector Predict(const Image& image);

  Vector GetWeights() const;
  void SetWeights(const Vector& weights);

  ModelType GetType() const { return type_; }
  void SetType(ModelType type) { type_ = type; }
  const Topology& GetTopology() const { return topology_; }

 private:
  Vector ExpectedOutput(const Image& image);

  Config config_;
  ModelType type_;
  Topology topology_;
  std::unique_ptr<EmnistParser> parser_;
  std::unique_ptr<Interface> mlp_;
  std::vector<Image> train_;
  std::vector<Image> test_;
  Metrics metrics_;
};

}  // namespace s21

#endif  // MLP_MODEL_MLP_H_
