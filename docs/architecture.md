# FL System Architecture

## Overview

This document describes the architecture of the Multimodal Federated Learning Security Framework.

## System Components

### 1. FL Server

The central server coordinates the federated learning process:

- **Aggregation**: Combines client model updates using FedAvg (or robust alternatives)
- **Model Distribution**: Sends global model to participating clients
- **Logging**: Tracks training progress and metrics
- **Defense Integration**: Applies defense mechanisms during aggregation

### 2. FL Clients

Each client represents a distributed data owner:

- **Local Training**: Trains on private local data
- **Model Exchange**: Sends updates to server, receives global model
- **Attack Simulation**: Can inject malicious updates (for research)
- **Evaluation**: Computes local accuracy and loss

### 3. Attack Module

Implements various attacks for security research:

- Label Flipping: Corrupts training labels
- Backdoor Triggers: Injects hidden patterns
- Model Poisoning: Scales updates maliciously

### 4. Defense Module

Implements defense mechanisms:

- Robust Aggregation: Krum, Trimmed Mean, Median
- Trust-Based: FLTrust with root dataset
- Privacy-Preserving: Differential Privacy

## Communication Protocol

```
Client 0 ─┐
Client 1 ─┼──► Server (FedAvg) ──► Global Model ──► All Clients
Client 2 ─┘
```

Each round:
1. Server sends global model parameters to clients
2. Clients train locally for E epochs
3. Clients send model updates to server
4. Server aggregates updates (FedAvg or robust method)
5. Repeat for R rounds

## Data Partitioning

### IID (Independent and Identically Distributed)
- Random uniform split across clients
- Each client has similar class distribution

### Non-IID (Dirichlet Distribution)
- Heterogeneous class distribution per client
- Controlled by alpha parameter (lower = more skewed)
- More realistic for real-world FL scenarios

## Configuration

All parameters are configurable via YAML files in `configs/`:

- Server settings (rounds, min clients)
- Client settings (local epochs, learning rate)
- Data settings (dataset, partition strategy)
- Attack settings (type, malicious clients)
- Defense settings (type, parameters)
