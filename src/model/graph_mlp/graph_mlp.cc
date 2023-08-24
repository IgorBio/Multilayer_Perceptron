#include "graph_mlp.h"

namespace s21 {

GraphMlp::GraphMlp(const Topology& topology) {
  net_.clear();

  net_.emplace_back(std::make_shared<Layer>(topology.GetInputSize()));

  for (std::size_t i = 1; i <= topology.GetHiddenCount(); ++i) {
    auto new_layer =
        std::make_shared<Layer>(topology.GetLayerSize(i), net_[i - 1]);
    net_.emplace_back(new_layer);
    net_[i - 1]->SetNextLayer(new_layer);
  }

  auto output_layer =
      std::make_shared<Layer>(topology.GetOutputSize(), net_.back());
  net_.emplace_back(output_layer);
  net_[net_.size() - 2]->SetNextLayer(output_layer);
}

void GraphMlp::SetInputLayer(const Vector& input_values) {
  if (input_values.size() != net_[0]->GetSize()) {
    throw std::invalid_argument(
        "Input values size doesn't match input layer size");
  }
  net_[0]->SetValues(input_values);
}

void GraphMlp::ForwardPropagation() {
  for (std::size_t i = 1; i < net_.size(); ++i) {
    net_[i]->FeedForward();
  }
}

void GraphMlp::BackPropagation(const Vector& expected, double learning_rate) {
  net_.back()->CalculateOutputError(expected);

  for (int i = net_.size() - 2; i >= 0; --i) {
    net_[i]->CalculateError();
  }

  for (std::size_t i = net_.size() - 1; i > 0; --i) {
    net_[i]->UpdateWeights(learning_rate);
  }
}

Vector GraphMlp::GetOutput() const {
  Vector output;
  auto& output_layer = net_.back()->GetLayer();
  for (std::size_t i = 0; i < output_layer.size(); ++i) {
    output.push_back(output_layer[i].GetValue());
  }

  return output;
}

std::pair<const Tensor, const Tensor> GraphMlp::GetMlp() const {
  Tensor weights, biases;

  for (std::size_t i = 1; i < net_.size(); ++i) {
    Matrix layer_weights;
    Matrix layer_biases(1);

    for (Neuron& neuron : net_[i]->GetLayer()) {
      layer_weights.emplace_back(neuron.GetWeights());
      layer_biases[0].emplace_back(neuron.GetBias());
    }

    weights.emplace_back(std::move(Transpose(layer_weights)));
    biases.emplace_back(std::move(layer_biases));
  }

  return {weights, biases};
}

void GraphMlp::SetMlp(const Tensor& weights, const Tensor& biases) {
  net_.clear();

  net_.emplace_back(std::make_shared<Layer>(weights[0].size()));

  for (std::size_t i = 1; i < weights.size(); ++i) {
    net_.emplace_back(std::make_shared<Layer>(weights[i].size(), net_[i - 1]));
    net_[i - 1]->SetNextLayer(net_[i]);
  }

  net_.emplace_back(
      std::make_shared<Layer>(weights.back()[0].size(), net_.back()));
  net_[net_.size() - 2]->SetNextLayer(net_.back());

  for (std::size_t i = 0; i < net_.size() - 1; ++i) {
    Matrix m = Transpose(weights[i]);
    for (std::size_t j = 0; j < net_[i + 1]->GetSize(); ++j) {
      net_[i + 1]->GetLayer()[j].SetWeights(m[j]);
      net_[i + 1]->GetLayer()[j].SetBias(biases[i][0][j]);
    }
  }
}

}  // namespace s21
