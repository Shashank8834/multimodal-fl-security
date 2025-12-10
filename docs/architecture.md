# Multimodal Federated Learning Security Framework
## System Architecture Document

### 1. Overview

This document describes the architecture of our Multimodal Federated Learning (MMFL) Security Framework, designed to study adversarial attacks and defense mechanisms in distributed multimodal learning systems.

---

### 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FL SERVER                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Aggregation Layer                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │   │
│  │  │ FedAvg   │ │  Krum    │ │ Trimmed  │ │   FLTrust    │   │   │
│  │  │(Baseline)│ │          │ │   Mean   │ │              │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│  ┌───────────────────────────┴────────────────────────────────┐    │
│  │                    Defense Module                           │    │
│  │  • Anomaly Detection  • Trust Scoring  • DP Noise Addition │    │
│  └────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                    Global Model Distribution                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   CLIENT 1    │    │   CLIENT 2    │    │   CLIENT N    │
│   (Benign)    │    │  (Malicious)  │    │   (Benign)    │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ Local Dataset │    │ Local Dataset │    │ Local Dataset │
│  ┌─────────┐  │    │  ┌─────────┐  │    │  ┌─────────┐  │
│  │ Images  │  │    │  │ Images  │  │    │  │ Images  │  │
│  ├─────────┤  │    │  ├─────────┤  │    │  ├─────────┤  │
│  │  Text   │  │    │  │  Text   │  │    │  │  Text   │  │
│  └─────────┘  │    │  └─────────┘  │    │  └─────────┘  │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ Local Model   │    │ Local Model   │    │ Local Model   │
│ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │
│ │Multimodal │ │    │ │Multimodal │ │    │ │Multimodal │ │
│ │   CNN     │ │    │ │   CNN     │ │    │ │   CNN     │ │
│ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │
├───────────────┤    ├───────────────┤    ├───────────────┤
│               │    │ Attack Module │    │               │
│               │    │ ┌───────────┐ │    │               │
│               │    │ │ Backdoor  │ │    │               │
│               │    │ │ LabelFlip │ │    │               │
│               │    │ └───────────┘ │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

### 3. Communication Protocol

```
Round r:
  1. Server → Clients: Send global model weights W_r
  2. Clients: Train locally for E epochs
  3. Clients → Server: Send model updates ΔW_i
  4. Server: Aggregate updates using defense mechanism
  5. Server: Update global model W_{r+1}
```

**Protocol**: gRPC via Flower framework
**Port**: 8080 (default)
**Serialization**: NumPy arrays

---

### 4. Component Details

#### 4.1 FL Server (`src/server/`)
- `fl_server.py`: Basic FedAvg server
- `robust_server.py`: Server with defense integration

#### 4.2 FL Client (`src/client/`)
- `fl_client.py`: Standard FL client
- `malicious_client.py`: Client with attack capabilities

#### 4.3 Attack Module (`src/attacks/`)
| Attack           | Type  | Description               |
| ---------------- | ----- | ------------------------- |
| LabelFlip        | Data  | Flip source→target labels |
| Backdoor         | Data  | Inject trigger patterns   |
| ModelReplacement | Model | Scale to dominate         |
| AdaptiveKrum     | Model | Evade Krum defense        |
| CrossModal       | Data  | Multimodal trigger        |

#### 4.4 Defense Module (`src/defenses/`)
| Defense     | Type        | Description               |
| ----------- | ----------- | ------------------------- |
| Krum        | Aggregation | Select nearest to cluster |
| TrimmedMean | Aggregation | Remove outliers           |
| FLTrust     | Trust       | Server-side validation    |
| DP-SGD      | Privacy     | Gradient noise            |

---

### 5. Data Flow

```
┌────────────┐    Partition    ┌────────────┐
│  Dataset   │ ──────────────► │  Client i  │
│ (CUB-200)  │    (IID/Non-IID)│  Subset    │
└────────────┘                 └────────────┘
                                     │
                               Local Training
                                     │
                                     ▼
                              ┌────────────┐
                              │   Update   │
                              │    ΔW_i    │
                              └────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
              ┌──────────┐    ┌──────────┐    ┌──────────┐
              │ Defense  │    │ Defense  │    │ Defense  │
              │  Check   │    │  Check   │    │  Check   │
              └──────────┘    └──────────┘    └──────────┘
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                               Aggregation
                                     │
                                     ▼
                              ┌────────────┐
                              │  Global    │
                              │  Model     │
                              └────────────┘
```

---

### 6. Modality Handling

**Approach**: Complete Modality with Complete Labels (CMC-MMFL)

Each client has:
- All modalities (image + text/attributes)
- Complete labels for all samples
- Independent local training

```
Client Data Structure:
{
    "image": Tensor[B, C, H, W],
    "text": Tensor[B, seq_len] or attributes: Tensor[B, 312],
    "label": Tensor[B]
}
```

---

### 7. Experiment Configuration

```yaml
experiment:
  dataset: "cub200"
  num_clients: 10
  num_rounds: 50
  
attack:
  type: "backdoor"
  malicious_clients: [0, 1]
  
defense:
  type: "krum"
  num_malicious: 2
```

---

### 8. Technology Stack

| Component     | Technology   |
| ------------- | ------------ |
| FL Framework  | Flower 1.0+  |
| Deep Learning | PyTorch 2.0+ |
| Communication | gRPC         |
| Configuration | YAML         |
| Tracking      | TensorBoard  |
