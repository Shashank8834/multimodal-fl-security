# RunPod Experiment Setup Guide

Complete step-by-step guide for running Federated Learning Security experiments on RunPod GPU instances.

---

## Table of Contents

1. [RunPod Instance Setup](#1-runpod-instance-setup)
2. [Environment Preparation](#2-environment-preparation)
3. [Dataset Setup](#3-dataset-setup)
4. [Running Experiments](#4-running-experiments)
5. [Estimated Runtimes](#5-estimated-runtimes)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. RunPod Instance Setup

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Create an account and add credits

### Step 2: Launch a GPU Instance
1. Click **"Deploy"** → **"GPU Pods"**
2. Select a template:
   - **Recommended**: `RunPod Pytorch 2.1` or `Jupyter Notebook`
3. Choose GPU:
   - **Minimum**: RTX 3080 (10GB VRAM) - sufficient for MNIST
   - **Recommended**: RTX 4090 (24GB VRAM) - required for CUB-200
   - **Budget option**: RTX 3090 (24GB VRAM)
4. Set container disk to at least **20GB**
5. Click **"Deploy"**

### Step 3: Access Jupyter
1. Wait for pod to start (1-2 minutes)
2. Click **"Connect"** → **"Jupyter Lab"**
3. Open a terminal in Jupyter

---

## 2. Environment Preparation

### Step 1: Clone the Repository

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/multimodal-fl-security.git
cd multimodal-fl-security
```

### Step 2: Install Dependencies

```bash
pip install -q flwr>=1.5.0 torch torchvision numpy pandas matplotlib seaborn scikit-learn pyyaml tqdm
```

### Step 3: Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output: `CUDA: True, GPU: NVIDIA GeForce RTX ...`

---

## 3. Dataset Setup

### MNIST (Auto-download)

MNIST downloads automatically when you run experiments. No manual steps needed.

```bash
# Test MNIST download
python -c "from torchvision import datasets; datasets.MNIST('./data', download=True); print('MNIST OK')"
```

### CUB-200 (Manual Download Required)

CUB-200-2011 must be downloaded manually.

#### Option A: Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Create Kaggle credentials (one-time setup)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download CUB-200
cd /workspace/multimodal-fl-security/data
kaggle datasets download -d wenewone/cub-200-2011
unzip cub-200-2011.zip
rm cub-200-2011.zip
```

#### Option B: Direct Download

1. Go to [Kaggle CUB-200](https://www.kaggle.com/datasets/wenewone/cub-200-2011)
2. Download `archive.zip`
3. Upload to RunPod via Jupyter file browser
4. Extract:

```bash
cd /workspace/multimodal-fl-security/data
unzip archive.zip
```

#### Verify CUB-200

```bash
ls -la /workspace/multimodal-fl-security/data/CUB_200_2011/
# Should see: images/, image_class_labels.txt, train_test_split.txt, etc.
```

---

## 4. Running Experiments

### Open Notebooks
1. In Jupyter Lab, navigate to `notebooks/` folder
2. Open notebooks in order (01 → 06)

### Execution Order

| Notebook                       | Purpose            | Time       |
| ------------------------------ | ------------------ | ---------- |
| `01_environment_setup.ipynb`   | Verify environment | 2 min      |
| `02_dataset_exploration.ipynb` | Explore datasets   | 5 min      |
| `03_quick_experiment.ipynb`    | Quick test         | 5-10 min   |
| `04_mnist_experiments.ipynb`   | Full MNIST         | 4-6 hours  |
| `05_cub200_experiments.ipynb`  | Full CUB-200       | 8-12 hours |
| `06_results_analysis.ipynb`    | Analyze results    | 10 min     |

### Quick Test First!

Run `03_quick_experiment.ipynb` first to verify everything works before starting long experiments.

### Running Long Experiments

For experiments taking hours:
1. Use **tmux** or **screen** to keep session alive
2. Or run from terminal instead of notebook:

```bash
cd /workspace/multimodal-fl-security
python experiments/run_paper_experiments.py --dataset mnist --seeds 1  # Quick test
python experiments/run_paper_experiments.py --dataset mnist            # Full MNIST
python experiments/run_paper_experiments.py --dataset cub200           # Full CUB-200
```

---

## 5. Estimated Runtimes

### MNIST Experiments
| Configuration                       | Time Estimate |
| ----------------------------------- | ------------- |
| Quick test (1 seed, 3 rounds)       | 15-20 min     |
| Single seed (5 seeds commented out) | 1-2 hours     |
| Full (5 seeds)                      | 4-6 hours     |

### CUB-200 Experiments
| Configuration       | Time Estimate |
| ------------------- | ------------- |
| Quick test (1 seed) | 30-60 min     |
| Full (3 seeds)      | 8-12 hours    |

### Tips to Save Time
1. Use `SEEDS = [42]` for initial testing
2. Reduce `num_rounds` for faster tests
3. Run MNIST first (faster) to validate pipeline

---

## 6. Troubleshooting

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` (try 8 or 4)
- Use fewer clients for CUB-200
- Clear GPU cache: `torch.cuda.empty_cache()`
- Restart kernel

### Module Not Found

```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
```python
import sys
sys.path.insert(0, '/workspace/multimodal-fl-security')
```

### CUB-200 Not Found

```
RuntimeError: CUB-200 dataset not found
```

**Solution:** Follow CUB-200 download instructions in Section 3.

### Kernel Crashes

If Jupyter kernel crashes during long runs:
1. Use terminal instead of notebook
2. Use tmux: `tmux new -s experiment`
3. Run: `python experiments/run_paper_experiments.py`
4. Detach: `Ctrl+B, D`
5. Reattach later: `tmux attach -t experiment`

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Results Location

After experiments complete, find results in:

```
experiments/
├── mnist_results_YYYYMMDD_HHMMSS/
│   ├── mnist_results.json      # Raw results
│   └── mnist_tables.tex        # LaTeX tables
├── cub200_results_YYYYMMDD_HHMMSS/
│   ├── cub200_results.json
│   └── cub200_tables.tex
└── combined_results.csv        # All results combined
```

---

## Quick Reference Commands

```bash
# Navigate to project
cd /workspace/multimodal-fl-security

# Install dependencies
pip install -q flwr torch torchvision numpy pandas matplotlib seaborn scikit-learn tqdm

# Quick MNIST test
python experiments/run_paper_experiments.py --quick

# Full MNIST (5 seeds)
python experiments/run_paper_experiments.py --dataset mnist

# Full CUB-200 (3 seeds)
python experiments/run_paper_experiments.py --dataset cub200

# Check GPU
nvidia-smi
```

---

## Support

For issues with this codebase, check:
- `PROJECT_KNOWLEDGE.md` - Full technical documentation
- `README.md` - Project overview
