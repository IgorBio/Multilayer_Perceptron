#include <gtest/gtest.h>

#include "../model/mlp.h"

using namespace s21;

TEST(MLP, Initialization1) {
  Topology topology = {1, 2, 2, 1};
  MLP mlp(topology);
  Vector weights = mlp.GetWeights();
  EXPECT_EQ(weights.size(), 6);
}

TEST(MLP, Initialization2) {
  Topology topology;
  MLP mlp(topology);
  Vector weights = mlp.GetWeights();
  EXPECT_EQ(weights.size(), 252000);
}

// TEST(MLP, XOR) {
//   // Create a dataset for XOR operation
//   Dataset train;
//   train.push_back({{0, 0}, {0}});
//   train.push_back({{0, 1}, {1}});
//   train.push_back({{1, 0}, {1}});
//   train.push_back({{1, 1}, {0}});

//   // Set up MLP
//   Topology topology = {2, 2, 1};
//   MLP mlp(topology);
//   mlp.SetTrainDataset(train);
//   mlp.SetEpochs(10000);
//   mlp.SetLearningRate(0.1);

//   // Train MLP on XOR dataset
//   mlp.Train();

//   // Test predictions on XOR dataset
//   for (const auto& example : train) {
//     Vector input = example.first;
//     Vector output = example.second;
//     Vector prediction = mlp.Predict(input);
//     ASSERT_EQ(prediction.size(), output.size());
//     // Check that predicted output is close enough to expected output
//     ASSERT_NEAR(prediction[0], output[0], 0.1);
//   }
// }

// TEST(MLP, LearningRate) {
//   // Create a dataset for AND operation
//   Dataset train;
//   train.push_back({{0, 0}, {0}});
//   train.push_back({{0, 1}, {0}});
//   train.push_back({{1, 0}, {0}});
//   train.push_back({{1, 1}, {1}});

//   // Set up MLP
//   Topology topology = {2, 2, 1};
//   MLP mlp(topology);
//   mlp.SetTrainDataset(train);
//   mlp.SetEpochs(10000);

//   // Train MLP on AND dataset with different learning rates
//   double learning_rates[] = {0.01, 0.1, 1};
//   for (double rate : learning_rates) {
//     mlp.SetLearningRate(rate);
//     auto start_time = std::chrono::system_clock::now();
//     mlp.Train();
//     auto end_time = std::chrono::system_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//         end_time - start_time);
//     std::cout << "Training time with rate " << rate << " is "
//               << duration.count() << " ms\n";
//   }
// }

// TEST(MLP, ActivationThreshold) {
//   // Create a dataset for XOR operation
//   Dataset train;
//   train.push_back({{0, 0}, {0}});
//   train.push_back({{0, 1}, {1}});
//   train.push_back({{1, 0}, {1}});
//   train.push_back({{1, 1}, {0}});

//   // Set up MLP
//   Topology topology = {2, 2, 1};
//   MLP mlp(topology);
//   mlp.SetTrainDataset(train);

//   // Train MLP with default activation threshold
//   mlp.SetEpochs(10000);
//   mlp.Train();
//   Metrics metrics = mlp.Test();
//   ASSERT_NEAR(metrics.accuracy, 0.75,
//               0.1);  // Expected accuracy for XOR dataset

//   // Train MLP with higher activation threshold
//   mlp.SetActivateThreshold(0.9);
//   mlp.Train();
//   metrics = mlp.Test();
//   ASSERT_NEAR(metrics.accuracy, 1, 0.1);  // Expected accuracy for XOR
//   dataset
// }