#include <gtest/gtest.h>

#include "../model/mlp.h"

using namespace s21;

TEST(MLP, Initialization1) {
  Topology topology{2, 2, 1};
  MLP mlp(topology);
  std::vector<Matrix> weights = mlp.GetWeights();
  EXPECT_EQ(weights.size(), 2);
}

TEST(MLP, Initialization2) {
  Topology topology;
  MLP mlp(topology);
  std::vector<Matrix> weights = mlp.GetWeights();
  EXPECT_EQ(weights.size(), 3);
}

TEST(MLP, XOR) {
  Dataset train;
  train.push_back({{0, 0}, {0}});
  train.push_back({{0, 1}, {1}});
  train.push_back({{1, 0}, {1}});
  train.push_back({{1, 1}, {0}});

  Topology topology{2, 4, 2};
  MLP mlp(topology);
  mlp.SetTrainDataset(train);
  mlp.SetEpochs(3000);
  mlp.SetLearningRate(0.1);

  mlp.Train();

  for (const auto& example : train) {
    Vector input = example.GetPixels();
    std::size_t output = example.GetLabel();
    std::size_t predict = mlp.PredictLabel(example);
    EXPECT_NEAR(predict, output, 0.1);
  }
}
