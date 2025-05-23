# ☁️ Jax TPU v6e: End-to-End Example

This repository provides a comprehensive guide and all necessary scripts to run a JAX-based Multi-Layer Perceptron (MLP) on a Google Cloud TPU. It covers environment setup, Artifact Registry configuration, Docker image creation, TPU VM provisioning, and execution of the training script.

## Overview

The primary goal of this project is to demonstrate a complete workflow for training a neural network (specifically, an MLP using Flax NNX) on Google Cloud TPUs. This includes:
*   Setting up your Google Cloud environment.
*   Packaging the JAX application into a Docker container.
*   Managing Docker images with Google Artifact Registry.
*   Provisioning and interacting with TPU VMs.
*   Running the training workload on the TPU.

This project is ideal for anyone looking to understand the practical steps involved in deploying JAX applications to Google Cloud TPUs.

## Project Structure

```
.
├── Dockerfile
├── JAX_MLP_EXPLAINED.md
├── README.md
├── jax_mlp_flax_nnx.py
└── jax_mlp_flax_nnx_colab.ipynb
```

## Key Files

*   **[`jax_mlp_flax_nnx.py`](jax_mlp_flax_nnx.py:1):** The core Python script containing the JAX MLP model definition (using Flax NNX), data generation, training loop, and evaluation logic.
*   **[`Dockerfile`](Dockerfile:1):** Defines the Docker image for running the JAX application, including all necessary dependencies.
*   **[`JAX_MLP_EXPLAINED.md`](JAX_MLP_EXPLAINED.md:1):** A detailed explanation of the `jax_mlp_flax_nnx.py` script, including the rationale behind package choices.
*   **[`jax_mlp_flax_nnx_colab.ipynb`](jax_mlp_flax_nnx_colab.ipynb:1):** A Google Colab notebook version of the script, adapted for easy execution and experimentation on Colab TPUs.
*   **[`README.md`](README.md:1):** This file – providing setup instructions and an overview of the project.

## Prerequisites

Before you begin, ensure you have the following installed and configured:
*   **Google Cloud SDK (`gcloud`)**: [Installation Guide](https://cloud.google.com/sdk/docs/install)
    *   Ensure you are authenticated: `gcloud auth login`
    *   Set your default project: `gcloud config set project YOUR_PROJECT_ID`
*   **Git**: [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
*   **(Optional but Recommended) GitHub CLI (`gh`)**: [Installation Guide](https://github.com/cli/cli#installation)
    *   Ensure you are authenticated: `gh auth login`

## Setup and Workflow

**Important:** Replace placeholder values like `YOUR_PROJECT_ID` with your actual details in the commands below or by setting the environment variables.

### 1. Configure Environment Variables

Define these variables in your shell session or at the beginning of a script for convenience.

```bash
# --- Core Configuration ---
export PROJECT_ID="YOUR_PROJECT_ID"       # Replace with your actual Google Cloud Project ID
export REGION="us-central1"             # Choose a region for Artifact Registry & TPUs (e.g., us-central1)
export ZONE="${REGION}-b"                 # Choose a zone within the region (e.g., us-central1-b)

# --- Artifact Registry Configuration ---
export AR_REPO_NAME="roo-jax-repo"        # Choose a name for your Artifact Registry repository
export DOCKER_IMAGE_NAME="jax-mlp-tpu"    # Name for your Docker image
export DOCKER_IMAGE_TAG="latest"
export AR_DOCKER_IMAGE_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"

# --- TPU Configuration ---
export TPU_NAME="roo-v6e-8"               # Name for your TPU VM
export ACCELERATOR_TYPE="v6e-8"         # TPU accelerator type (e.g., v6e-8, v4-8)
export TPU_RUNTIME_VERSION="v2-alpha-tpuv6e" # TPU runtime version (check available versions)
```

### 2. Create Artifact Registry Repository (One-time Setup)

If you haven't already, create a Docker repository in Google Artifact Registry.

```bash
gcloud artifacts repositories create "${AR_REPO_NAME}" \
  --project="${PROJECT_ID}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Docker repository for JAX MLP images"
```
*   Run `gcloud artifacts locations list --project="${PROJECT_ID}"` to see available locations for Artifact Registry.

### 3. Create TPU VM (One-time Setup, or as needed)

Provision a TPU VM for your training workload.

```bash
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --version="${TPU_RUNTIME_VERSION}"
```

### 4. Build and Push Docker Image

Package your application into a Docker image and push it to your Artifact Registry.

```bash
gcloud builds submit . \
  --project="${PROJECT_ID}" \
  --tag="${AR_DOCKER_IMAGE_PATH}"
```

### 5. Run Training on TPU VM

SSH into the TPU VM and execute the Docker container.

```bash
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=all \
  --command=" \
    echo 'Attempting to configure Docker for Artifact Registry...' && \
    sudo gcloud auth configure-docker ${REGION}-docker.pkg.dev -q && \
    echo 'Docker configured. Attempting to pull image...' && \
    sudo docker pull ${AR_DOCKER_IMAGE_PATH} && \
    echo 'Image pulled. Attempting to run container...' && \
    sudo docker run --rm --shm-size=1g ${AR_DOCKER_IMAGE_PATH} && \
    echo 'Container execution finished.' \
  "
```
*   `--worker=all`: Ensures the command runs on all workers if it's a Pod. For single TPUs, it targets the main worker.
*   `--shm-size=1g`: Allocates shared memory to the Docker container, which can be important for some JAX/XLA operations.

### Example Successful Output

```
WARNING:absl:Using 'flax.experimental.nnx' is deprecated. Please use 'flax.nnx' instead.
X_train shape: (400, 4), y_train_one_hot shape: (400, 3)
X_test shape: (100, 4), y_test_one_hot shape: (100, 3)
Number of classes: 3

Starting training for 100 epochs...
Epoch 1/100, Avg Loss: 1.2975
Epoch 10/100, Avg Loss: 1.3104
Epoch 20/100, Avg Loss: 1.2985
Epoch 30/100, Avg Loss: 1.3112
Epoch 40/100, Avg Loss: 1.2920
Epoch 50/100, Avg Loss: 1.2994
Epoch 60/100, Avg Loss: 1.3115
Epoch 70/100, Avg Loss: 1.2941
Epoch 80/100, Avg Loss: 1.3070
Epoch 90/100, Avg Loss: 1.3098
Epoch 100/100, Avg Loss: 1.3041
Training finished.

Test Accuracy: 0.2600

```
### 6. (Optional) List TPUs

To check the status and details of your TPU VMs:

```bash
gcloud compute tpus tpu-vm list --project="${PROJECT_ID}" --zone="${ZONE}"
```

## Important Notes

*   **Permissions**: Ensure the service account used by your TPU VM (or your user account if running commands locally) has the necessary IAM permissions (e.g., Artifact Registry Reader, Compute Instance Admin (v1) for TPU operations, etc.).
*   **Costs**: Be mindful of the costs associated with Google Cloud services, especially TPUs and Artifact Registry storage. Stop or delete resources when not in use.
    *   Delete TPU VM: `gcloud compute tpus tpu-vm delete "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}"`
    *   Delete Artifact Registry Repository (be cautious!): `gcloud artifacts repositories delete "${AR_REPO_NAME}" --project="${PROJECT_ID}" --location="${REGION}"`
*   **`sudo` for Docker**: The `sudo` prefix for Docker commands on the TPU VM is used as per common practice on these VMs where the default user might not be in the `docker` group.
*   **TPU Runtime Versions**: TPU runtime versions (`--version` flag during TPU creation) are updated periodically. Refer to the [official Google Cloud documentation](https://cloud.google.com/tpu/docs/supported-versions) for the latest supported versions for JAX.

---

Happy VIBE Coding on GCP! 🚀