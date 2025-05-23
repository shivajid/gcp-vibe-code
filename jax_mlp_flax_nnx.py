import jax
import jax.numpy as jnp
import flax.experimental.nnx as nnx
import optax
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import functools

# 1. MLP Definition using Flax NNX
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

# 2. Synthetic Data Generation
def generate_synthetic_data(n_samples=200, n_features=2, n_classes=2, random_state=42):
    """Generates simple synthetic data for classification."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state
    )
    y_one_hot = jax.nn.one_hot(y, num_classes=n_classes)
    return jnp.array(X), jnp.array(y_one_hot), jnp.array(y)

# 3. Training Loop Components
# Loss Function (Cross-Entropy for classification)
def cross_entropy_loss(logits, labels_one_hot):
    """Computes cross-entropy loss."""
    return -jnp.sum(labels_one_hot * jax.nn.log_softmax(logits), axis=-1).mean()

# Optimizer
def get_optimizer(learning_rate=1e-3):
    return optax.adam(learning_rate)

# Training Step Function
@functools.partial(jax.jit, static_argnames=('loss_fn', 'optimizer_update_fn'))
def train_step(params, static, opt_state, loss_fn, optimizer_update_fn, X_batch, y_batch):
    """Performs a single training step with Flax NNX using split state."""
    def loss_for_grad(current_params):
        # Reconstruct the model for the forward pass
        model_for_forward_pass = nnx.merge(current_params, static)
        logits = model_for_forward_pass(X_batch) # Call the model directly
        return loss_fn(logits, y_batch)

    loss_val, grads = jax.value_and_grad(loss_for_grad)(params) # Grads w.r.t. params
    updates, new_opt_state = optimizer_update_fn(grads, opt_state, params) # Pass params for optimizer
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val

# Prediction function
@jax.jit
def predict(params, static, X):
    """Makes predictions using the model with split state."""
    model_for_prediction = nnx.merge(params, static)
    return model_for_prediction(X)


if __name__ == "__main__":
    # Configuration
    N_SAMPLES = 500
    N_FEATURES = 4
    N_CLASSES = 3
    HIDDEN_SIZES = [64, 32]
    LEARNING_RATE = 0.005
    EPOCHS = 100
    BATCH_SIZE = 32
    PRINT_EVERY_EPOCHS = 10
    SEED = 42

    key = jax.random.PRNGKey(SEED)
    key_data, key_init, key_shuffle = jax.random.split(key, 3)

    # Generate Data
    X_data, y_data_one_hot, y_data_orig = generate_synthetic_data(
        n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES, random_state=SEED
    )
    X_train, X_test, y_train_one_hot, y_test_one_hot, y_train_orig, y_test_orig = train_test_split(
        X_data, y_data_one_hot, y_data_orig, test_size=0.2, random_state=SEED
    )

    print(f"X_train shape: {X_train.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"X_test shape: {X_test.shape}, y_test_one_hot shape: {y_test_one_hot.shape}")
    print(f"Number of classes: {N_CLASSES}")

    # Initialize Model and Optimizer with Flax NNX
    # Create a dummy input to infer shapes
    dummy_x = X_train[:1]

    # NNX model initialization
    # The model is stateful. We initialize it and it becomes the state.
    model = MLP(input_size=N_FEATURES, hidden_sizes=HIDDEN_SIZES, output_size=N_CLASSES, rngs=nnx.Rngs(params=key_init))
    
    # Initialize parameters by a "dry run"
    # This ensures the model is built and shapes are inferred.
    _ = model(dummy_x) 

    # Split the model into trainable parameters and static parts
    params, static = nnx.split(model)

    optimizer = get_optimizer(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params) # Optimizer initializes with trainable parameters

    # Training Loop
    num_train_samples = X_train.shape[0]
    num_batches = num_train_samples // BATCH_SIZE

    print(f"\nStarting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        key_shuffle, key_loop = jax.random.split(key_shuffle)
        permutation = jax.random.permutation(key_loop, num_train_samples)
        shuffled_X_train = X_train[permutation]
        shuffled_y_train_one_hot = y_train_one_hot[permutation]

        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            X_batch = shuffled_X_train[start_idx:end_idx]
            y_batch = shuffled_y_train_one_hot[start_idx:end_idx]

            params, opt_state, loss_val = train_step(
                params,         # Current trainable parameters
                static,         # Static part of the model
                opt_state,
                cross_entropy_loss, # loss_fn
                optimizer.update,   # optimizer_update_fn
                X_batch,
                y_batch
            )
            epoch_loss += loss_val

        avg_epoch_loss = epoch_loss / num_batches
        if (epoch + 1) % PRINT_EVERY_EPOCHS == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_epoch_loss:.4f}")

    print("Training finished.")

    # Evaluation (simple accuracy)
    # For prediction, pass params and static to the predict function, which handles the merge.
    test_logits = predict(params, static, X_test)
    predicted_classes = jnp.argmax(test_logits, axis=1)
    accuracy = jnp.mean(predicted_classes == y_test_orig)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nTo run this code:")
    print("1. Ensure you have JAX, Flax (for NNX), Optax, and scikit-learn installed:")
    print("   pip install jax jaxlib flax optax scikit-learn")
    print("   (Note: flax.experimental.nnx is part of the main flax package)")
    print("2. Save the code as a Python file (e.g., jax_mlp_flax_nnx.py).")
    print("3. Run from the terminal: python jax_mlp_flax_nnx.py")
    print("\nNote: For TPU training, ensure your JAX installation is configured for TPUs.")
    print("The use of jax.jit helps in compiling functions for efficient execution on accelerators.")