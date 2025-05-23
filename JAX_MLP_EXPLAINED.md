# Explanation of the JAX MLP Script (`src/jax_mlp_flax_nnx.py`)

This document provides a detailed explanation of the Python script [`src/jax_mlp_flax_nnx.py`](src/jax_mlp_flax_nnx.py:1), which implements and trains a Multi-Layer Perceptron (MLP) using JAX, Flax NNX, and Optax.

## Core Functionality

The script performs the following key operations:
1.  **Defines an MLP model**: Uses the Flax NNX API for a more stateful and Pythonic way of defining neural network layers.
2.  **Generates Synthetic Data**: Creates a simple classification dataset using `scikit-learn`.
3.  **Sets up Training Components**: Defines a loss function (cross-entropy) and an optimizer (Adam).
4.  **Implements a Training Loop**: Trains the MLP model on the synthetic data, handling model state (parameters vs. static parts) explicitly for compatibility with JAX transformations and Optax.
5.  **Evaluates the Model**: Calculates the accuracy of the trained model on a test set.

## Code Breakdown

### 1. Imports and Setup

```python
import jax
import jax.numpy as jnp
import flax.experimental.nnx as nnx # Or flax.nnx in newer versions
import optax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import functools
```
*   [`jax`](src/jax_mlp_flax_nnx.py:1) and [`jax.numpy`](src/jax_mlp_flax_nnx.py:2) (`jnp`): Core libraries for high-performance numerical computing, automatic differentiation (`jax.grad`), and JIT compilation (`jax.jit`).
*   [`flax.experimental.nnx`](src/jax_mlp_flax_nnx.py:3): The experimental (now stable as `flax.nnx`) Flax Neural Network eXperimental API. It provides a more object-oriented and stateful way to define models compared to the "functional core" style of traditional Flax. This makes managing model parameters and static attributes more explicit.
*   [`optax`](src/jax_mlp_flax_nnx.py:4): A gradient processing and optimization library for JAX. It provides a wide range of optimizers (like Adam, SGD) and tools for manipulating gradients.
*   [`sklearn.datasets.make_classification`](src/jax_mlp_flax_nnx.py:5): Used to generate a synthetic dataset for a classification task, allowing for controlled experiments.
*   [`sklearn.model_selection.train_test_split`](src/jax_mlp_flax_nnx.py:6): A utility to split the dataset into training and testing sets.
*   [`functools`](src/jax_mlp_flax_nnx.py:7): Used for `functools.partial` to pre-fill arguments for the `jax.jit` decorator, making the `train_step` function cleaner.

### 2. MLP Definition (`MLP` class)

```python
class MLP(nnx.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, *, rngs: nnx.Rngs):
        current_in_features = input_size
        self.hidden_layers = []
        for h_size in hidden_sizes:
            layer = nnx.Linear(in_features=current_in_features, out_features=h_size, rngs=rngs)
            self.hidden_layers.append(layer)
            current_in_features = h_size
        
        self.output_layer = nnx.Linear(in_features=current_in_features, out_features=output_size, rngs=rngs)

    def __call__(self, x: jax.Array):
        for layer in self.hidden_layers:
            x = layer(x)
            x = jax.nn.relu(x)
        x = self.output_layer(x)
        return x
```
*   Inherits from [`nnx.Module`](src/jax_mlp_flax_nnx.py:10), the base class for NNX models.
*   The `__init__` method constructs the layers:
    *   It iteratively creates [`nnx.Linear`](src/jax_mlp_flax_nnx.py:14) layers for the hidden part of the network.
    *   An [`nnx.Rngs`](src/jax_mlp_flax_nnx.py:11) object is passed to initialize layer parameters. NNX makes PRNG key handling more explicit at the module level.
*   The `__call__` method defines the forward pass:
    *   Data `x` flows through each hidden layer followed by a ReLU activation ([`jax.nn.relu`](src/jax_mlp_flax_nnx.py:23)).
    *   The final output layer produces the logits.

### 3. Synthetic Data Generation (`generate_synthetic_data`)

```python
def generate_synthetic_data(n_samples=200, n_features=2, n_classes=2, random_state=42):
    # ... uses make_classification ...
    y_one_hot = jax.nn.one_hot(y, num_classes=n_classes)
    return jnp.array(X), jnp.array(y_one_hot), jnp.array(y)
```
*   Uses [`make_classification`](src/jax_mlp_flax_nnx.py:30) from `scikit-learn` for simplicity and reproducibility.
*   Converts labels `y` to one-hot encoding using [`jax.nn.one_hot`](src/jax_mlp_flax_nnx.py:40) as this is often required for categorical cross-entropy loss.
*   Returns JAX arrays (`jnp.array`).

### 4. Training Loop Components

*   **Loss Function (`cross_entropy_loss`)**:
    ```python
    def cross_entropy_loss(logits, labels_one_hot):
        return -jnp.sum(labels_one_hot * jax.nn.log_softmax(logits), axis=-1).mean()
    ```
    A standard cross-entropy loss implementation suitable for multi-class classification. It uses [`jax.nn.log_softmax`](src/jax_mlp_flax_nnx.py:46) for numerical stability.

*   **Optimizer (`get_optimizer`)**:
    ```python
    def get_optimizer(learning_rate=1e-3):
        return optax.adam(learning_rate)
    ```
    Returns an Adam optimizer instance from [`optax.adam`](src/jax_mlp_flax_nnx.py:50).

*   **Training Step (`train_step`)**:
    ```python
    @functools.partial(jax.jit, static_argnames=('loss_fn', 'optimizer_update_fn'))
    def train_step(params, static, opt_state, loss_fn, optimizer_update_fn, X_batch, y_batch):
        def loss_for_grad(current_params):
            model_for_forward_pass = nnx.merge(current_params, static) # Reconstruct model
            logits = model_for_forward_pass(X_batch)
            return loss_fn(logits, y_batch)

        loss_val, grads = jax.value_and_grad(loss_for_grad)(params) # Get loss and gradients
        updates, new_opt_state = optimizer_update_fn(grads, opt_state, params) # Get updates from optimizer
        new_params = optax.apply_updates(params, updates) # Apply updates
        return new_params, new_opt_state, loss_val
    ```
    *   **JIT Compilation**: Decorated with [`jax.jit`](src/jax_mlp_flax_nnx.py:53) for performance. `loss_fn` and `optimizer_update_fn` are marked as `static_argnames` because their Python functions are part of the computation graph's structure, not dynamic JAX array inputs.
    *   **NNX State Handling**:
        *   NNX models are split into trainable `params` (e.g., weights, biases) and `static` parts (e.g., layer structure, non-trainable attributes) using [`nnx.split()`](src/jax_mlp_flax_nnx.py:103).
        *   Inside `loss_for_grad`, the model is temporarily reconstructed using [`nnx.merge(current_params, static)`](src/jax_mlp_flax_nnx.py:57) to perform the forward pass. This is crucial because JAX transformations like `jax.grad` operate on functions of JAX arrays (the `params`).
    *   **Gradient Calculation**: [`jax.value_and_grad`](src/jax_mlp_flax_nnx.py:61) computes both the loss and the gradients with respect to `params`.
    *   **Optimizer Update**: The optimizer's `update` function ([`optimizer.update`](src/jax_mlp_flax_nnx.py:130)) calculates parameter updates from gradients and the current optimizer state.
    *   **Parameter Update**: [`optax.apply_updates`](src/jax_mlp_flax_nnx.py:63) applies these updates to the parameters.

*   **Prediction Function (`predict`)**:
    ```python
    @jax.jit
    def predict(params, static, X):
        model_for_prediction = nnx.merge(params, static)
        return model_for_prediction(X)
    ```
    Similar to `train_step`, it merges `params` and `static` to reconstruct the model for making predictions. JIT-compiled for speed.

### 5. Main Execution Block (`if __name__ == "__main__":`)

*   **Configuration**: Sets hyperparameters like learning rate, epochs, batch size, etc.
*   **PRNG Keys**: Initializes JAX PRNG keys ([`jax.random.PRNGKey`](src/jax_mlp_flax_nnx.py:81)) and splits them for different purposes (data generation, model initialization, shuffling). JAX requires explicit PRNG key management.
*   **Data Preparation**:
    *   Calls [`generate_synthetic_data`](src/jax_mlp_flax_nnx.py:85).
    *   Splits data into training and test sets using [`train_test_split`](src/jax_mlp_flax_nnx.py:88).
*   **Model Initialization**:
    *   An MLP instance is created: `model = MLP(...)`.
    *   A "dry run" `_ = model(dummy_x)` ([`src/jax_mlp_flax_nnx.py:100`](src/jax_mlp_flax_nnx.py:100)) is performed. This is a common pattern in NNX (and stateful Flax) to ensure all layers are built and parameters are initialized before they are accessed or split.
    *   The model is split: `params, static = nnx.split(model)` ([`src/jax_mlp_flax_nnx.py:103`](src/jax_mlp_flax_nnx.py:103)).
*   **Optimizer Initialization**: `opt_state = optimizer.init(params)` ([`src/jax_mlp_flax_nnx.py:106`](src/jax_mlp_flax_nnx.py:106)). The optimizer state is initialized based on the model's trainable parameters.
*   **Training Loop**:
    *   Iterates for a specified number of `EPOCHS`.
    *   Shuffles training data in each epoch using [`jax.random.permutation`](src/jax_mlp_flax_nnx.py:114).
    *   Iterates through batches of data.
    *   Calls the `train_step` function to update model parameters and optimizer state.
    *   Prints average loss periodically.
*   **Evaluation**:
    *   Calls the `predict` function on the test set.
    *   Calculates accuracy by comparing predicted classes ([`jnp.argmax`](src/jax_mlp_flax_nnx.py:140)) with true labels.

## Rationale for Package Choices

*   **JAX (`jax`, `jax.numpy`)**:
    *   **High Performance**: JAX can compile Python functions (via XLA) to run very efficiently on accelerators like GPUs and TPUs.
    *   **Automatic Differentiation**: [`jax.grad`](src/jax_mlp_flax_nnx.py:61) is fundamental for training neural networks, as it automatically computes gradients of the loss function with respect to model parameters.
    *   **Functional Programming Paradigm**: JAX encourages a functional style (pure functions), which simplifies reasoning about code and is well-suited for transformations like JIT compilation and parallelization (`jax.pmap`, `jax.vmap`).
    *   **NumPy API**: `jax.numpy` provides a familiar NumPy-like API, making it easier for users with NumPy experience to adopt JAX.

*   **Flax (`flax.experimental.nnx` or `flax.nnx`)**:
    *   **Neural Network Library for JAX**: Flax is the primary neural network library for JAX, providing tools for building and training models.
    *   **NNX API**: The NNX API was chosen here because it offers a more stateful, object-oriented approach to defining models.
        *   **Explicit State Management**: Parameters (`params`) and static graph structure (`static`) are explicitly managed using [`nnx.split`](src/jax_mlp_flax_nnx.py:103) and [`nnx.merge`](src/jax_mlp_flax_nnx.py:57). This makes it clearer what parts of the model are trainable and how state is handled within JAX's functional paradigm, especially when working with optimizers like Optax that expect parameters as explicit inputs.
        *   **Pythonic Feel**: For developers accustomed to PyTorch or Keras, NNX can feel more intuitive as model layers are attributes of the model class.
        *   **Compatibility with JAX Transformations**: Despite being stateful, NNX is designed to work seamlessly with `jax.jit`, `jax.grad`, etc., by providing these mechanisms to separate and re-merge state.

*   **Optax**:
    *   **Dedicated Optimization Library**: While JAX provides `jax.grad`, it doesn't include built-in optimizers. Optax fills this gap.
    *   **Rich Set of Optimizers**: Offers a comprehensive collection of standard optimizers (Adam, SGD, RMSProp, etc.) and learning rate schedules.
    *   **Gradient Transformations**: Allows for complex gradient manipulations (clipping, masking, etc.) in a composable way.
    *   **Stateful Optimizers**: Optax optimizers are stateful (e.g., Adam needs to track momentum), and Optax manages this state explicitly, which integrates well with JAX's functional nature. The `optimizer.init` and `optimizer.update` pattern is standard.

*   **Scikit-learn (`sklearn`)**:
    *   **Data Utilities**: `scikit-learn` is a mature and widely used machine learning library in Python.
    *   **Synthetic Data Generation**: [`make_classification`](src/jax_mlp_flax_nnx.py:30) is a convenient way to create simple, reproducible datasets for testing and demonstration without needing external data files.
    *   **Data Splitting**: [`train_test_split`](src/jax_mlp_flax_nnx.py:88) is a standard utility for dividing data into training and testing sets, crucial for evaluating model generalization.
    *   While JAX could be used for these, `scikit-learn` provides these off-the-shelf, saving development time for these common preprocessing tasks.

This combination of libraries provides a powerful and flexible environment for developing and training high-performance neural networks in Python, with JAX at the core for computation, Flax NNX for model definition, Optax for optimization, and Scikit-learn for data utilities.