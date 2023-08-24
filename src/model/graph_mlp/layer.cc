#include "layer.h"

namespace s21 {

Layer::Layer(std::size_t size, std::shared_ptr<Layer> prev)
    : layer_(size), prev_layer_(prev), next_layer_(nullptr) {
  for (Neuron& neuron : layer_) {
    neuron = Neuron(prev ? prev->GetSize() : 0);
  }
}

void Layer::SetValues(const Vector& values) {
  if (values.size() != layer_.size()) {
    throw std::invalid_argument("Input size doesn't match layer size");
  }

  for (std::size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].SetValue(values[i]);
  }
}

void Layer::FeedForward() {
  for (Neuron& neuron : layer_) {
    neuron.CalculateValue(GetPrevValues());
  }
}

void Layer::CalculateOutputError(const Vector& expected) {
  if (expected.size() != layer_.size()) {
    throw std::invalid_argument(
        "Expected output size doesn't match layer size");
  }

  std::size_t idx = std::distance(
      expected.begin(), std::find(expected.begin(), expected.end(), 1.0));
  for (std::size_t i = 0; i < layer_.size(); ++i) {
    double value = layer_[i].GetValue();
    double target = (i == idx) ? 1.0 : 0.0;
    layer_[i].CalculateError(target - value);
  }
}

void Layer::CalculateError() {
  for (std::size_t i = 0; i < layer_.size(); ++i) {
    layer_[i].CalculateError(ErrorSum(i));
  }
}

void Layer::UpdateWeights(double learning_rate) {
  if (prev_layer_) {
    for (Neuron& neuron : layer_) {
      neuron.UpdateWeights(GetPrevValues(), learning_rate);
    }
  }
}

double Layer::ErrorSum(std::size_t idx) const {
  double sum = 0.0;

  if (next_layer_) {
    for (std::size_t i = 0; i < next_layer_->GetLayer().size(); ++i) {
      sum += next_layer_->GetLayer()[i].GetError() *
             next_layer_->GetLayer()[i].GetWeight(idx);
    }
  }

  return sum;
}

Vector Layer::GetPrevValues() const {
  Vector prev_values;
  if (prev_layer_) {
    auto& prev_layer = prev_layer_->GetLayer();
    prev_values.reserve(prev_layer.size());

    for (Neuron& neuron : prev_layer) {
      prev_values.push_back(neuron.GetValue());
    }
  }

  return prev_values;
}

}  // namespace s21
