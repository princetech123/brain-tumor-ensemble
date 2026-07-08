# Installation and Deployment Guide

This guide details steps to install, run, customize, and verify the Multi-Model Ensemble framework.

---

## Prerequisites

Ensure your host machine has the following tools installed:
- Python 3.9, 3.10, or 3.11
- Git
- Docker and Docker Compose (Optional, for containerized deployment)
- A CUDA-compatible GPU with drivers (Optional, for accelerated training)

---

## Local Installation (Recommended for Development)

### 1. Repository Setup
```bash
git clone https://github.com/your-username/brain_tumor_ensemble.git
cd brain_tumor_ensemble
```

### 2. Environment Configuration
Create and activate a virtual environment to isolate project packages:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Package Installation
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: If your host requires specific PyTorch CUDA installations, configure them prior to running pip.*

### 4. Create Directory Structures
Create the workspace directories:
```bash
mkdir -p checkpoints logs data src/web/static/uploads src/web/static/temp src/web/static/eval
```

---

## Containerized Deployment (Recommended for Production)

### 1. Build and Run Compose Services
Execute the following orchestration command from the root folder:
```bash
docker-compose up --build
```
This builds the `neuro_ensemble_app` image, mounts the local `checkpoints/`, `logs/`, and `data/` directories, and hosts the Flask dashboard on port `5001`.

### 2. Verify Container Execution
Ensure the service is running:
```bash
docker ps
```
Open [http://localhost:5001](http://localhost:5001) in your browser.

---

## Running the Complete System Flow

Follow these execution steps:

### Step 1: Create Dataset
Generate the synthetic train/val/test MRI splits locally:
```bash
python3 generate_synthetic_data.py
```

### Step 2: Validate Components
Run the unit test blocks:
```bash
python3 -m unittest discover tests/
```

### Step 3: Run Training
Fine-tune the three backbone networks and optimize the attention fusion network:
```bash
python3 src/training.py
```

### Step 4: Launch Web Dashboard
Start the web dashboard to upload images and visualize predictions:
```bash
python3 src/web/app.py
```
Open [http://localhost:5001](http://localhost:5001) in your browser.

---

## Troubleshooting Common Errors

### 1. ImportError: `libGL.so.1: cannot open shared object file`
- **Cause:** OpenCV requires GLIB/OpenGL packages which are missing from slim Linux environments.
- **Solution (Ubuntu/Debian):**
  ```bash
  sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
  ```
- *Note: This is automatically resolved when deploying via the provided `Dockerfile`.*

### 2. CUDA Out of Memory (OOM)
- **Cause:** High batch size or complex transformer models saturating GPU memory.
- **Solution:** Reduce the batch size or input resolution in `config/config.yaml`.
  ```yaml
  data:
    batch_size: 8 # Decreased from 16
  ```

### 3. Checkpoints Missing Warning
- **Cause:** Launching `app.py` before executing `training.py`.
- **Solution:** The server will run with default weights as a fallback. For diagnostic accuracy, execute the training pipeline first:
  ```bash
  python3 src/training.py
  ```
