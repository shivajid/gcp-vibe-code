# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY jax_mlp_flax_nnx.py /app/

# Install JAX with TPU support, Flax, Optax, and scikit-learn
# This command fetches the appropriate jaxlib version for TPUs.
RUN pip install --no-cache-dir \
    jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    flax \
    optax \
    scikit-learn

# Define environment variable (optional)
ENV NAME JAXMLP_TPU

# Run jax_mlp_flax_nnx.py when the container launches
CMD ["python", "jax_mlp_flax_nnx.py"]