# Multimodal Federated Learning Security Framework

A research framework for studying security vulnerabilities and defenses in Multimodal Federated Learning (MMFL) systems.

## Overview

This project implements a comprehensive framework for:
- **Federated Learning**: Distributed training using the Flower framework
- **Attack Simulation**: Label-flipping, backdoor triggers, model poisoning
- **Defense Mechanisms**: Krum, Trimmed Mean, FLTrust, Differential Privacy
- **Multimodal Learning**: Image + Text modality fusion

## Project Structure

```
multimodal-fl-security/
├── src/
│   ├── client/          # FL client implementation
│   ├── server/          # FL server with aggregation strategies
│   ├── attacks/         # Attack implementations
│   ├── defenses/        # Defense mechanisms
│   ├── models/          # Neural network models
│   └── utils/           # Utilities (data loading, metrics, etc.)
├── data/                # Datasets (MNIST, CUB-200)
├── configs/             # Experiment configurations
├── experiments/         # Logs and results
├── docs/                # Documentation
└── tests/               # Unit tests
```

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic FL Training (MNIST)

**Terminal 1 - Start Server:**
```bash
python src/server/fl_server.py --num-rounds 10 --min-clients 3
```

**Terminal 2, 3, 4 - Start Clients:**
```bash
python src/client/fl_client.py --client-id 0
python src/client/fl_client.py --client-id 1
python src/client/fl_client.py --client-id 2
```

### 3. Run with Configuration File

```bash
python run_experiment.py --config configs/default.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FL Server                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Aggregation Strategy (FedAvg/Krum/TrimmedMean) │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
│                         │ Model Updates                 │
│         ┌───────────────┼───────────────┐              │
│         │               │               │              │
│         ▼               ▼               ▼              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │ Client 0 │    │ Client 1 │    │ Client 2 │         │
│  │ (Benign) │    │(Malicious│    │ (Benign) │         │
│  └──────────┘    └──────────┘    └──────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Implemented Components

### Attacks
- [ ] Label Flipping Attack
- [ ] Backdoor Trigger Attack
- [ ] Model Replacement Attack
- [ ] Adaptive Krum Attack

### Defenses
- [ ] Krum Aggregation
- [ ] Multi-Krum
- [ ] Trimmed Mean
- [ ] Coordinate-wise Median
- [ ] FLTrust
- [ ] Differential Privacy (DP-SGD)

### Datasets
- [x] MNIST (baseline testing)
- [x] CUB-200-2011 (fine-grained bird classification)

## Configuration

Example configuration file (`configs/default.yaml`):

```yaml
experiment:
  name: "baseline_fedavg"
  seed: 42

server:
  num_rounds: 50
  min_clients: 3
  aggregation: "fedavg"

client:
  num_clients: 10
  local_epochs: 5
  batch_size: 32
  learning_rate: 0.01

data:
  dataset: "mnist"
  partition: "iid"  # or "noniid"
  num_classes: 10

attack:
  enabled: false
  type: "label_flip"
  malicious_clients: [0, 1]
  poison_ratio: 0.1

defense:
  enabled: false
  type: "krum"
```

## Team

- **Shashank**: Architecture Design & Framework Setup
- **Dravid**: Attack Literature & Implementation
- **Siddharth**: Defense Literature & Strategy
- **Shreyas**: Dataset Preparation & Experiments

## License

MIT License - For Research Purposes Only
