use nalgebra::DMatrix;
use rand::Rng;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

// Import the dataset module (src/dataset.rs)
mod dataset;

use serde::{Deserialize, Serialize};

/// Sigmoid activation function: 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid (for hidden layers).
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// A neural network layer containing weights and biases.
/// - `weights` is a (input_size × output_size) matrix.
/// - `biases` is a 1×output_size row vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    weights: DMatrix<f64>,
    biases: DMatrix<f64>,
}

impl Layer {
    /// Create a new layer with weights initialized randomly (range: -0.1 to 0.1)
    /// and biases set to zero.
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = DMatrix::from_fn(input_size, output_size, |_, _| {
            rng.gen_range(-0.1..0.1)
        });
        let biases = DMatrix::zeros(1, output_size);
        Self { weights, biases }
    }
}

/// Adam optimizer state and hyperparameters.
#[derive(Serialize, Deserialize)]
struct Adam {
    learning_rate: f64,
    beta1: f64,   // Momentum decay
    beta2: f64,   // RMSProp decay
    epsilon: f64, // To avoid division by zero
    m_weights: Vec<DMatrix<f64>>,
    v_weights: Vec<DMatrix<f64>>,
    m_biases: Vec<DMatrix<f64>>,
    v_biases: Vec<DMatrix<f64>>,
    t: f64,       // Timestep
}

impl Adam {
    /// Initialize Adam given a slice of layer sizes (e.g. [784, 256, 128, 64, 10]).
    fn new(learning_rate: f64, sizes: &[usize]) -> Self {
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();
        for i in 0..sizes.len() - 1 {
            m_weights.push(DMatrix::zeros(sizes[i], sizes[i + 1]));
            v_weights.push(DMatrix::zeros(sizes[i], sizes[i + 1]));
            m_biases.push(DMatrix::zeros(1, sizes[i + 1]));
            v_biases.push(DMatrix::zeros(1, sizes[i + 1]));
        }
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            t: 0.0,
        }
    }

    /// Update a given layer’s parameters using the Adam algorithm.
    /// The layer index is used to update the proper moment estimates.
    fn update(
        &mut self,
        layer: &mut Layer,
        weight_gradients: &DMatrix<f64>,
        bias_gradients: &DMatrix<f64>,
        layer_idx: usize,
    ) {
        self.t += 1.0;
        // Update first moment estimates.
        let m_w = &self.m_weights[layer_idx] * self.beta1 + weight_gradients * (1.0 - self.beta1);
        let m_b = &self.m_biases[layer_idx] * self.beta1 + bias_gradients * (1.0 - self.beta1);
        // Update second moment estimates.
        let v_w = &self.v_weights[layer_idx] * self.beta2
            + weight_gradients.component_mul(weight_gradients) * (1.0 - self.beta2);
        let v_b = &self.v_biases[layer_idx] * self.beta2
            + bias_gradients.component_mul(bias_gradients) * (1.0 - self.beta2);
        // Bias-corrected moment estimates (clone to avoid moves).
        let m_w_hat = m_w.clone() * (1.0 / (1.0 - self.beta1.powf(self.t)));
        let v_w_hat = v_w.clone() * (1.0 / (1.0 - self.beta2.powf(self.t)));
        let m_b_hat = m_b.clone() * (1.0 / (1.0 - self.beta1.powf(self.t)));
        let v_b_hat = v_b.clone() * (1.0 / (1.0 - self.beta2.powf(self.t)));
        // Compute updates.
        let weight_updates = m_w_hat.zip_map(&v_w_hat, |m, v| {
            self.learning_rate * m / (v.sqrt() + self.epsilon)
        });
        layer.weights -= weight_updates;
        let bias_updates = m_b_hat.zip_map(&v_b_hat, |m, v| {
            self.learning_rate * m / (v.sqrt() + self.epsilon)
        });
        layer.biases -= bias_updates;
        // Saving updated moments.
        self.m_weights[layer_idx] = m_w;
        self.v_weights[layer_idx] = v_w;
        self.m_biases[layer_idx] = m_b;
        self.v_biases[layer_idx] = v_b;
    }

    /// Resize the optimizer,s moment estimates after the network grows.
    fn resize(&mut self, sizes: &[usize]) {
        self.m_weights.clear();
        self.v_weights.clear();
        self.m_biases.clear();
        self.v_biases.clear();
        for i in 0..sizes.len() - 1 {
            self.m_weights.push(DMatrix::zeros(sizes[i], sizes[i + 1]));
            self.v_weights.push(DMatrix::zeros(sizes[i], sizes[i + 1]));
            self.m_biases.push(DMatrix::zeros(1, sizes[i + 1]));
            self.v_biases.push(DMatrix::zeros(1, sizes[i + 1]));
        }
    }
}

/// The neural network  
///dynamic growth by adding neurons or layers.
#[derive(Serialize, Deserialize)]
struct NeuralNetwork {
    layers: Vec<Layer>,
    max_hidden_neurons: usize,
    max_hidden_layers: usize,
}

impl NeuralNetwork {
    /// Create a new network from a vector of layer sizes.
    fn new(sizes: Vec<usize>, max_hidden_neurons: usize, max_hidden_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }
        Self {
            layers,
            max_hidden_neurons,
            max_hidden_layers,
        }
    }

    /// Performs a forward pass through the network.
    /// Returns a vector of activations (one per layer, including the input).
    fn forward(&self, input: DMatrix<f64>) -> Vec<DMatrix<f64>> {
        let mut activations = vec![input.clone()];
        let mut current = input;
        for layer in &self.layers {
            current = (current * &layer.weights + &layer.biases).map(sigmoid);
            activations.push(current.clone());
        }
        activations
    }

    /// Dynamically adds a neuron to the specified layer.
    fn add_neuron(&mut self, layer_idx: usize) {
        if layer_idx >= self.layers.len() {
            return;
        }
        let current_neurons = self.layers[layer_idx].weights.ncols();
        if current_neurons >= self.max_hidden_neurons {
            println!(
                "Max hidden neurons ({}) reached in layer {}, skipping.",
                self.max_hidden_neurons, layer_idx
            );
            return;
        }
        let mut rng = rand::thread_rng();
        let input_size = self.layers[layer_idx].weights.nrows();
        let new_neurons = current_neurons + 1;
        // Expand current layer's weights.
        let mut new_weights = DMatrix::<f64>::zeros(input_size, new_neurons);
        new_weights
            .slice_mut((0, 0), (input_size, current_neurons))
            .copy_from(&self.layers[layer_idx].weights);
        for i in 0..input_size {
            new_weights[(i, current_neurons)] = rng.gen_range(-0.1..0.1);
        }
        self.layers[layer_idx].weights = new_weights;
        // Expand biases.
        let mut new_biases = DMatrix::<f64>::zeros(1, new_neurons);
        new_biases
            .slice_mut((0, 0), (1, current_neurons))
            .copy_from(&self.layers[layer_idx].biases);
        self.layers[layer_idx].biases = new_biases;
        // If a subsequent layer exists, add a new row.
        if layer_idx + 1 < self.layers.len() {
            let output_size = self.layers[layer_idx + 1].weights.ncols();
            let mut new_next_weights = DMatrix::<f64>::zeros(new_neurons, output_size);
            new_next_weights
                .slice_mut((0, 0), (current_neurons, output_size))
                .copy_from(&self.layers[layer_idx + 1].weights);
            for j in 0..output_size {
                new_next_weights[(current_neurons, j)] = rng.gen_range(-0.1..0.1);
            }
            self.layers[layer_idx + 1].weights = new_next_weights;
        }
        println!(
            "Added neuron to layer {}. New neuron count: {}",
            layer_idx, new_neurons
        );
    }

    /// Dynamically adds a new hidden layer.
    fn add_layer(&mut self, neuron_count: usize) {
        if self.layers.len() - 1 >= self.max_hidden_layers {
            println!(
                "Max hidden layers ({}) reached, skipping.",
                self.max_hidden_layers
            );
            return;
        }
        let mut rng = rand::thread_rng();
        let last_hidden_idx = self.layers.len() - 2;
        let input_size = self.layers[last_hidden_idx].biases.ncols();
        let output_size = self.layers[last_hidden_idx + 1].weights.ncols();
        let new_layer = Layer::new(input_size, neuron_count);
        let new_output_layer = Layer::new(neuron_count, output_size);
        self.layers.insert(last_hidden_idx + 1, new_layer);
        self.layers[last_hidden_idx + 2] = new_output_layer;
        println!(
            "Added hidden layer with {} neurons. Total layers: {}",
            neuron_count,
            self.layers.len()
        );
    }

    /// Train the network with the given inputs and targets.
    /// Uses cross-entropy loss with the Adam optimizer.
    /// If loss improvement stalls, dynamic growth is triggered.
    fn train(
        &mut self,
        inputs: Vec<DMatrix<f64>>,
        targets: Vec<DMatrix<f64>>,
        epochs: usize,
        optimizer: &mut Adam,
    ) {
        let mut prev_loss = f64::INFINITY;
        let patience = 50;
        let mut patience_counter = 0;
        let loss_threshold = 0.001;

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let activations = self.forward(input.clone());
                let output = activations.last().unwrap();
                // Compute cross-entropy loss and error gradient.
                let mut error = DMatrix::<f64>::zeros(output.nrows(), output.ncols());
                for i in 0..output.len() {
                    let p = output[i].max(1e-8).min(1.0 - 1e-8);
                    let t = target[i];
                    total_loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
                    error[i] = p - t;
                }
                total_loss /= inputs.len() as f64;
                // Backpropagation.
                let mut delta = error.clone();
                for i in (0..self.layers.len()).rev() {
                    let prev_activation = &activations[i];
                    let weight_gradients = prev_activation.transpose() * &delta;
                    let bias_gradients = delta.clone();
                    optimizer.update(&mut self.layers[i], &weight_gradients, &bias_gradients, i);
                    if i > 0 {
                        delta = &delta * self.layers[i].weights.transpose();
                        let act = &activations[i];
                        delta = delta.component_mul(&act.map(sigmoid_derivative));
                    }
                }
            }

            // Trigger dynamic network growth if improvement is minimal.
            if (prev_loss - total_loss).abs() < loss_threshold {
                patience_counter += 1;
                if patience_counter >= patience {
                    if self.layers[0].biases.ncols() < self.max_hidden_neurons {
                        self.add_neuron(0);
                        let sizes: Vec<usize> = self
                            .layers
                            .iter()
                            .map(|l| l.weights.nrows())
                            .chain(std::iter::once(self.layers.last().unwrap().biases.ncols()))
                            .collect();
                        optimizer.resize(&sizes);
                    } else if self.layers.len() - 1 < self.max_hidden_layers {
                        self.add_layer(64);
                        let sizes: Vec<usize> = self
                            .layers
                            .iter()
                            .map(|l| l.weights.nrows())
                            .chain(std::iter::once(self.layers.last().unwrap().biases.ncols()))
                            .collect();
                        optimizer.resize(&sizes);
                    }
                    patience_counter = 0;
                }
            } else {
                patience_counter = 0;
            }
            prev_loss = total_loss;

            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoch {}: Loss = {}", epoch, total_loss);
            }
        }
    }
}

/// A checkpoint structure to save both the network and optimizer state.
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    network: NeuralNetwork,
    optimizer: Adam,
}

/// Save the checkpoint to a file.
fn save_checkpoint(checkpoint: &Checkpoint, path: &str) {
    let encoded: Vec<u8> = bincode::serialize(checkpoint).expect("Serialization failed");
    let mut file = fs::File::create(path).expect("Failed to create checkpoint file");
    file.write_all(&encoded).expect("Failed to write checkpoint");
    println!("Checkpoint saved to {}", path);
}

/// Attempt to load a checkpoint from a file.
fn load_checkpoint(path: &str) -> Option<Checkpoint> {
    if !Path::new(path).exists() {
        return None;
    }
    let mut file = fs::File::open(path).expect("Failed to open checkpoint file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read checkpoint file");
    let checkpoint: Checkpoint = bincode::deserialize(&buffer).expect("Deserialization failed");
    println!("Checkpoint loaded from {}", path);
    Some(checkpoint)
}

fn main() {
    // Try to load a checkpoint if it exists.
    let checkpoint_path = "checkpoint.bin";
    let (mut nn, mut optimizer) = if let Some(cp) = load_checkpoint(checkpoint_path) {
        (cp.network, cp.optimizer)
    } else {
        // Otherwise, create a new network and optimizer.
        // In this example: 784 inputs, hidden layers with 256, 128, 64 neurons, and 10 outputs.
        let nn = NeuralNetwork::new(vec![784, 256, 128, 64, 10], 512, 5);
        let optimizer = Adam::new(0.001, &[784, 256, 128, 64, 10]);
        (nn, optimizer)
    };

    // Load training data.
    // can use the simulated MNIST-like data or load an external dataset via dataset::load_dataset().
    // For example, to load external CSV data, uncomment the lines below:
    // let (inputs, targets) = dataset::load_dataset("mnist_data.csv").expect("Failed to load dataset");
    // For now we use simulated data:
    let (inputs, targets) = dataset::generate_mnist_like_data(1000);

    // Train the network.
    nn.train(inputs.clone(), targets.clone(), 1000, &mut optimizer);

    // Save a checkpoint so that training progress is not lost.
    let cp = Checkpoint { network: nn, optimizer };
    save_checkpoint(&cp, checkpoint_path);

    // Test the trained network on a few samples.
    println!("\nTesting the trained network:");
    for i in 0..5 {
        let activations = cp.network.forward(inputs[i].clone());
        let output = activations.last().unwrap();
        let predicted = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let target_label = targets[i]
            .iter()
            .position(|&x| (x - 1.0).abs() < 1e-6)
            .unwrap();
        println!("Sample {}: Predicted = {}, Target = {}", i, predicted, target_label);
    }
}
