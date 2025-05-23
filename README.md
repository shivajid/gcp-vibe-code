# JAX MLP on TPU Execution Commands

This file lists the commands used to create a Google Cloud TPU VM, set up an Artifact Registry, build the Docker image, push it to Artifact Registry, and run the JAX MLP training script.

**Important:** Replace `YOUR_PROJECT_ID` with your actual Google Cloud Project ID throughout these commands.

## 0. Setup Environment Variables

Define these variables at the beginning of your script or session.

```bash
export PROJECT_ID="YOUR_PROJECT_ID" # Replace with your actual project ID
export AR_REPO_NAME="roo-jax-repo"    # Choose a name for your Artifact Registry repository
export AR_LOCATION="us-central1"      # Choose a location for your Artifact Registry (e.g., us-central1)
export ACCELERATOR_TYPE="v6e-8"     # Example: v6e-8, v6e-16, etc.
export TPU_NAME="roo-v6e-8"
export TPU_ZONE="us-central1-b"
export TPU_VERSION="v2-alpha-tpuv6e"
```

## 1. Create Artifact Registry Repository (if it doesn't exist)

This command creates a new Docker repository in Google Artifact Registry.

```bash
gcloud artifacts repositories create ${AR_REPO_NAME} \
  --repository-format=docker \
  --location=${AR_LOCATION} \
  --description="Docker repository for JAX MLP images" \
  --project=${PROJECT_ID}
```
*   `AR_REPO_NAME`: The name you chose for your repository.
*   `AR_LOCATION`: The location for your repository.
*   Run `gcloud artifacts locations list` to see available locations.

## 2. Create TPU VM (v6e-8)

This command creates a new TPU v6e-8 VM.

```bash
gcloud compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${TPU_ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${TPU_VERSION} \
  --project=${PROJECT_ID}
```
*   `TPU_NAME`: Name for your TPU VM.
*   `TPU_ZONE`: Zone where the TPU VM will be created.
*   `ACCELERATOR_TYPE`: Specifies the TPU type and size.
*   `TPU_VERSION`: The runtime version for the TPU VM.
*   `PROJECT_ID`: Your Google Cloud project ID.

## 3. Build and Push Docker Image

This command builds the Docker image from the `Dockerfile` in the current directory, tags it, and pushes it to your Google Artifact Registry.

```bash
gcloud builds submit --tag ${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/jax-mlp-tpu:latest . --project=${PROJECT_ID}
```

## 4. List TPUs (Optional Debugging Step)

This command lists available TPUs in the specified zone and project.

```bash
gcloud compute tpus list --zone=${TPU_ZONE} --project=${PROJECT_ID}
```

## 5. SSH into TPU VM and Run Docker Container

This command SSHes into the specified TPU VM and executes a series of commands on the VM:
1.  Authenticates Docker with Google Artifact Registry (`sudo gcloud auth configure-docker ${AR_LOCATION}-docker.pkg.dev`).
2.  Pulls the latest version of the Docker image (`sudo docker pull ${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/jax-mlp-tpu:latest`).
3.  Runs the Docker container (`sudo docker run --rm ${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/jax-mlp-tpu:latest`).

```bash
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --project=${PROJECT_ID} \
  --zone=${TPU_ZONE} \
  --command="sudo gcloud auth configure-docker ${AR_LOCATION}-docker.pkg.dev && sudo docker pull ${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/jax-mlp-tpu:latest && sudo docker run --rm ${AR_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO_NAME}/jax-mlp-tpu:latest"
```

## Notes:
*   Ensure you have the Google Cloud SDK installed and configured.
*   Set the environment variables at the beginning, replacing `YOUR_PROJECT_ID` with your actual project ID.
*   The `sudo` prefix for Docker commands on the TPU VM is necessary.