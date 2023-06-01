#include "graph_mlp.h"

namespace s21 {

GraphMlp::GraphMlp(Topology topology) : net_{topology.GetLayersCount()} {
  AddLayer(topology.GetInputSize(), topology.GetLayerSize(1));

  for (std::size_t i{1u}; i < topology.GetHiddenCount(); ++i) {
    AddLayer(topology.GetLayerSize(i), topology.GetLayerSize(i + 1));
  }

  AddLayer(topology.GetLastHidden(), topology.GetOutputSize());
}

void GraphMlp::AddLayer(std::size_t input_size, std::size_t output_size) {
  net_.emplace_back(std::make_unique<Layer>(input_size, output_size));
}

void GraphMlp::SetInputLayer(const Vector& input) {
  Layer& input_layer = *net_.front();
  for (std::size_t i{0u}; i < input_layer.GetLayer().size(); ++i)
    input_layer.GetLayer()[i]->SetValue(input[i]);
}

Vector GraphMlp::Predict(const Vector& input) {
  Vector output = input;
  for (const auto& layer : net_) {
    for (const auto& neuron : layer->GetLayer()) {
      double sum{0.0};
      const Vector& weights{neuron->GetWeights()};
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

double GraphMlp::CalculateLoss(const Vector& predicted_output,
                               const Vector& expected_output) {
  double loss = 0.0;
  for (std::size_t i{0u}; i < predicted_output.size(); ++i) {
    double diff = expected_output[i] - predicted_output[i];
    loss += diff * diff;
  }
  return loss;
}

// void GraphMlp::ForwardPropagation() {
//   for (std::size_t i{1u}; i < net_.size(); ++i) {
//     auto& prev_layer = *net_[i - 1];
//     auto& curr_layer = *net_[i];

// #pragma omp parallel for
//     for (std::size_t j = 0u; j < curr_layer.GetLayer().size(); ++j) {
//       auto& curr_neuron = *curr_layer.GetLayer()[j];
//       double sum{0.0};
//       const auto& weights = curr_neuron.GetWeights();

//       for (std::size_t k{0u}; k < prev_layer.GetLayer().size(); ++k) {
//         auto& prev_neuron = *prev_layer.GetLayer()[k];
//         sum += prev_neuron.GetValue() * weights[k];
//       }
//       curr_neuron.SetValue(ApplyActivation(sum, sigmoid));
//     }
//   }
// }

void GraphMlp::ForwardPropagation() {
  // Iterate over each layer in the network, passing the output of the previous
  // layer to the input of the current layer
  for (std::size_t i = 1; i < net_.size(); ++i) {
    Layer& prev_layer = *net_[i - 1];
    Layer& curr_layer = *net_[i];
    Matrix layer_output(curr_layer.GetLayer().size(),
                        Vector(prev_layer.GetLayer().size()));
    // Iterate over each neuron in the current layer and calculate its output
    // using the output of the previous layer as input
    for (std::size_t j = 0; j < curr_layer.GetLayer().size(); ++j) {
      Neuron& curr_neuron = *curr_layer.GetLayer()[j];
      double total_input = 0.0;
      // Iterate over each neuron in the previous layer and multiply its output
      // by the corresponding weight for this neuron in the current layer
      for (std::size_t k = 0; k < prev_layer.GetLayer().size(); ++k) {
        Neuron& prev_neuron = *prev_layer.GetLayer()[k];
        total_input += prev_neuron.GetValue() * curr_neuron.GetWeights()[k];
      }
      // Apply the activation function to the total input to get the output of
      // the current neuron
      double output = ApplyActivation(total_input, &sigmoid);
      curr_neuron.SetValue(output);
      // Store the output of the current neuron for the next layer to use as
      // input
      layer_output[j] = Vector{output};
    }
    // Set the output of the current layer to be the output of all its neurons
    curr_layer.SetLayer(layer_output);
  }
}

void GraphMlp::BackPropagation(const Vector& expected_output,
                               double learning_rate) {
  // Calculate the output layer error
  Layer& output_layer = *net_.back();
  const Matrix& output_values = output_layer.GetValues();
  Matrix output_error = output_layer.CalculateLoss(expected_output);

  // Update the output layer weights
  output_layer.UpdateLayer(output_error, learning_rate);

  // Propagate the error backwards through the network
  for (std::size_t i{net_.size() - 2}; i > 0; --i) {
    Layer& current_layer = *net_[i];
    Layer& next_layer = *net_[i + 1];

    // Calculate the error for this layer
    const Matrix& current_values = current_layer.GetValues();
    const Matrix& next_weights = next_layer.GetWeights();
    Matrix current_error =
        current_layer.CalculateLoss(next_weights, next_layer.GetLayerError());

    // Update the weights for this layer
    current_layer.UpdateLayer(current_error, learning_rate);
  }
}

Vector GraphMlp::GetOutput() const {
  Vector output;
  Layer& output_layer = *net_.back();
  output.reserve(output_layer.GetLayer().size());
  std::transform(output_layer.GetLayer().cbegin(),
                 output_layer.GetLayer().cend(), std::back_inserter(output),
                 [](const auto& neuron) { return neuron->GetValue(); });
  return output;
}

Tensor GraphMlp::GetWeights() const {
  Tensor weights;
  for (const auto& layer : net_) {
    Matrix layer_weights;
    for (const auto& neuron : layer->GetLayer()) {
      Vector neuron_weights = neuron->GetWeights();
      layer_weights.push_back(neuron_weights);
    }
    weights.push_back(layer_weights);
  }
  return weights;
}

void GraphMlp::SetWeights(const Tensor& weights) {
  net_.resize(weights.size());
  for (std::size_t i{0u}; i < net_.size(); ++i) {
    Layer& layer = *net_[i];
    layer.SetLayer(weights[i]);
  }
}

}  // namespace s21
