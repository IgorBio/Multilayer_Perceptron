#include "graph_mlp.h"

namespace s21 {

GraphMlp::GraphMlp(Topology topology) : net_{topology.hidden_layers + 2} {
  AddLayer(topology.input_layer, topology.hidden_layer);

  for (std::size_t i{1u}; i < topology.hidden_layers; ++i) {
    AddLayer(topology.hidden_layer, topology.hidden_layer);
  }

  AddLayer(topology.hidden_layer, topology.output_layer);
}

void GraphMlp::AddLayer(std::size_t input_size, std::size_t output_size) {
  net_.emplace_back(std::make_unique<Layer>(input_size, output_size));
}

void GraphMlp::SetInputLayer(const Vector& input) {
  auto& input_layer = *net_.front();
  for (std::size_t i{0u}; i < input_layer.GetLayer().size(); ++i)
    input_layer.GetLayer()[i]->SetValue(input[i]);
}

Vector GraphMlp::Predict(const Vector& input) const {
  Vector output = input;
  for (const auto& layer : net_) {
    for (const auto& neuron : layer->GetLayer()) {
      double sum{neuron->GetBias()};
      const auto& weights{neuron->GetWeights()};
      for (std::size_t i{0u}; i < weights.size(); ++i) {
        sum += weights[i] * output[i];
      }
      neuron->SetValue(ApplyActivation(sum, sigmoid));
    }
    output.clear();
    for (const auto& neuron : layer->GetLayer()) {
      output.push_back(neuron->GetValue());
    }
  }
  return output;
}

double GraphMlp::CalculateLoss(const Matrix& inputs,
                               const Matrix& expected_outputs) {
  double total_loss{0.0};
#pragma omp parallel for reduction(+ : total_loss)
  for (std::size_t i{0u}; i < inputs.size(); ++i) {
    SetInputLayer(inputs[i]);

    ForwardPropagation();

    const auto output = GetOutput();
    const auto& expected_output = expected_outputs[i];

    for (std::size_t j{0u}; j < output.size(); ++j) {
      const auto diff = output[j] - expected_output[j];
      total_loss += (diff * diff);
    }
  }
  return total_loss / inputs.size();
}

void GraphMlp::ForwardPropagation() {
  for (std::size_t i{1u}; i < net_.size(); ++i) {
    auto& prev_layer = *net_[i - 1];
    auto& curr_layer = *net_[i];

#pragma omp parallel for
    for (std::size_t j{0u}; j < curr_layer.GetLayer().size(); ++j) {
      auto& curr_neuron = *curr_layer.GetLayer()[j];
      double sum{0.0};
      const auto& weights = curr_neuron.GetWeights();

      for (std::size_t k{0u}; k < prev_layer.GetLayer().size(); ++k) {
        auto& prev_neuron = *prev_layer.GetLayer()[k];
        sum += prev_neuron.GetValue() * weights[k];
      }
      sum += curr_neuron.GetBias();
      curr_neuron.SetValue(ApplyActivation(sum, sigmoid));
    }
  }
}

void GraphMlp::BackPropagation(const Vector& expected, double learning_rate) {
  auto output_layer = net_.back().get();
  auto output_errors = output_layer->CalculateLoss(expected);

  UpdateWeights(output_errors, learning_rate, net_.size() - 1);

  for (std::size_t i = net_.size() - 2; i > 0; --i) {
    auto& layer = *net_[i];
    const auto neurons = layer.GetLayer();

    Matrix next_errors(neurons.size(), Vector(neurons[0]->GetWeights().size()));

    for (std::size_t j = 0; j < neurons.size(); ++j) {
      auto& neuron = *neurons[j];
      Vector errors_with_activation(neurons[0]->GetWeights().size());
      for (std::size_t k = 0; k < neurons[0]->GetWeights().size(); ++k) {
        double error{0.0};
        for (std::size_t l = 0; l < net_[i + 1]->GetLayer().size(); ++l) {
          auto& next_neuron = *net_[i + 1]->GetLayer()[l];
          error += next_errors[l][k] * next_neuron.GetWeights()[j];
        }
        errors_with_activation[k] =
            error *
            ApplyActivationDerivative(neuron.GetValue(), sigmoid_derivative);
      }
      neuron.UpdateWeights(errors_with_activation, learning_rate);
      next_errors[j] = errors_with_activation;
    }

    UpdateWeights(next_errors, learning_rate, i);
  }

  auto& input_layer = *net_.front();
  const auto input_neurons = input_layer.GetLayer();
  Matrix next_errors(input_neurons.size(),
                     Vector(input_neurons[0]->GetWeights().size()));

  for (std::size_t i = 0; i < input_neurons.size(); ++i) {
    auto& neuron = *input_neurons[i];
    Vector errors_with_activation(input_neurons[0]->GetWeights().size());
    for (std::size_t j = 0; j < input_neurons[0]->GetWeights().size(); ++j) {
      double error{0.0};
      for (std::size_t k = 0; k < net_[1]->GetLayer().size(); ++k) {
        auto& next_neuron = *net_[1]->GetLayer()[k];
        error += next_errors[k][j] * next_neuron.GetWeights()[i];
      }
      errors_with_activation[j] =
          error *
          ApplyActivationDerivative(neuron.GetValue(), sigmoid_derivative);
    }
    neuron.UpdateWeights(errors_with_activation, learning_rate);
    next_errors[i] = errors_with_activation;
  }

  UpdateWeights(next_errors, learning_rate, 0);
}

void GraphMlp::UpdateWeights(const Matrix& errors, double learning_rate,
                             std::size_t idx) {
  net_[idx]->UpdateWeights(errors, learning_rate);
}

Vector GraphMlp::GetOutput() const {
  Vector output;
  const auto& output_layer = *net_.back();
  output.reserve(output_layer.GetLayer().size());
  std::transform(output_layer.GetLayer().cbegin(),
                 output_layer.GetLayer().cend(), std::back_inserter(output),
                 [](const auto& neuron) { return neuron->GetValue(); });
  return output;
}

Vector GraphMlp::GetWeights() const {
  Vector weights;
  for (const auto& layer : net_) {
    for (const auto& neuron : layer->GetLayer()) {
      const auto& neuron_weights = neuron->GetWeights();
      weights.insert(weights.end(), neuron_weights.cbegin(),
                     neuron_weights.cend());
      weights.push_back(neuron->GetBias());
    }
  }
  return weights;
}

void GraphMlp::SetWeights(const Vector& weights) {
  std::size_t index{0u};
  for (auto& layer : net_) {
    for (auto& neuron : layer->GetLayer()) {
      auto& neuron_weights = neuron->GetWeights();
      std::copy_n(weights.begin() + index, neuron_weights.size(),
                  neuron_weights.begin());
      index += neuron_weights.size();
      neuron->SetBias(weights[index++]);
    }
  }
}

}  // namespace s21