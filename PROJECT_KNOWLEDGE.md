# Multimodal Federated Learning Security - Complete Project Knowledge

This document provides an in-depth explanation of the entire project, covering Federated Learning fundamentals, all implemented attacks, all defenses, and the codebase architecture.

---

## Table of Contents

1. [Federated Learning Fundamentals](#federated-learning-fundamentals)
2. [Threat Model](#threat-model)
3. [Data Distributions: IID vs Non-IID](#data-distributions-iid-vs-non-iid)
4. [Attacks](#attacks)
   - [Data Poisoning Attacks](#data-poisoning-attacks)
   - [Model Poisoning Attacks](#model-poisoning-attacks)
5. [Defenses](#defenses)
6. [Metrics](#metrics)
7. [Codebase Architecture](#codebase-architecture)
8. [Experiment Pipeline](#experiment-pipeline)

---

## Federated Learning Fundamentals

### What is Federated Learning?

**Federated Learning (FL)** is a distributed machine learning paradigm where multiple clients (devices/institutions) collaboratively train a shared model without sharing their raw data.

```
┌─────────────────────────────────────────────────────────────┐
│                    CENTRAL SERVER                           │
│                   ┌─────────────┐                           │
│                   │   Global    │                           │
│                   │    Model    │                           │
│                   └─────────────┘                           │
│                         │                                   │
│        ┌────────────────┼────────────────┐                  │
│        ▼                ▼                ▼                  │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │Client 1 │     │Client 2 │     │Client N │              │
│   │ (Data)  │     │ (Data)  │     │ (Data)  │              │
│   └─────────┘     └─────────┘     └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### FedAvg Algorithm (Federated Averaging)

The standard FL algorithm, introduced by McMahan et al. (2017):

```
For each round t = 1, 2, ..., T:
    1. Server sends global model W_t to all clients
    2. Each client k:
       - Trains locally on their data for E epochs
       - Produces updated model W_t^k
    3. Server aggregates updates:
       W_{t+1} = Σ (n_k / n) * W_t^k
       where n_k = samples on client k, n = total samples
```

**Why FL?**
- **Privacy**: Raw data never leaves client devices
- **Communication**: Only model updates are transferred (not raw data)
- **Regulatory Compliance**: GDPR, HIPAA compliance for sensitive data
- **Scalability**: Can handle millions of edge devices

---

## Threat Model

### Attack Taxonomy

| Attack Type         | What is Poisoned | Attacker Goal                        | Stealthiness |
| ------------------- | ---------------- | ------------------------------------ | ------------ |
| **Data Poisoning**  | Training data    | Degrade accuracy or implant backdoor | Medium       |
| **Model Poisoning** | Model updates    | Manipulate aggregation               | Low to High  |

### Attacker Capabilities

1. **Byzantine Clients**: Some clients may be compromised or malicious
2. **Fraction Bound**: Typically assume f < n/3 malicious clients (Byzantine fault tolerance)
3. **Knowledge**: Attacker may or may not know other clients' updates

### Attack Goals

| Goal           | Description                    | Success Metric                 |
| -------------- | ------------------------------ | ------------------------------ |
| **Untargeted** | Reduce global model accuracy   | Low test accuracy              |
| **Targeted**   | Misclassify specific class(es) | High error on target class     |
| **Backdoor**   | Insert hidden trigger          | High Attack Success Rate (ASR) |

---

## Data Distributions: IID vs Non-IID

### IID (Independent and Identically Distributed)

All clients have data drawn from the same distribution.

```
Client 1: [0,1,2,3,4,5,6,7,8,9] ← All classes equally represented
Client 2: [0,1,2,3,4,5,6,7,8,9]
Client 3: [0,1,2,3,4,5,6,7,8,9]
```

**In practice**: Rare in real-world FL. Data on your phone is very different from others'.

### Non-IID (Heterogeneous)

Clients have different data distributions. This project uses **Dirichlet distribution** for partitioning.

```
α (alpha) controls heterogeneity:
- α = ∞  → IID (all clients have same distribution)
- α = 1.0 → Moderate heterogeneity
- α = 0.5 → High heterogeneity
- α = 0.1 → Extreme heterogeneity (each client has 1-2 classes)

Client 1: [0,0,0,1,1]         ← Mostly class 0, 1
Client 2: [3,3,4,4,4]         ← Mostly class 3, 4
Client 3: [7,8,8,9,9,9]       ← Mostly class 7, 8, 9
```

**Impact on FL**:
- Non-IID makes convergence harder
- Local models "drift" towards local distributions
- Increases importance of robust aggregation

---

## Attacks

### Data Poisoning Attacks

Data poisoning attacks corrupt the training data on malicious clients.

---

#### 1. Label Flip Attack

**File**: `src/attacks/label_flip.py`

**Concept**: Change labels of training samples to confuse the model.

**How it works**:
```
Original:  Image(digit 0) → Label: 0
Poisoned:  Image(digit 0) → Label: 8  (flipped!)
```

**Mathematical Formulation**:
```
For samples x with label y = source_class:
    With probability poison_ratio:
        y' = target_class
```

**Implementation Details**:
```python
class LabelFlipAttack:
    source_class = 0      # Flip samples labeled as 0
    target_class = 8      # Change their label to 8
    poison_ratio = 0.3    # Flip 30% of source class samples
```

**Impact**:
- Model accuracy decreases on source class
- Source class samples get misclassified as target
- Overall accuracy drops 5-15%

**Variants**:
- **AllToOneAttack**: Flip ALL labels to target class (more aggressive)

---

#### 2. Backdoor Attack

**File**: `src/attacks/backdoor.py`

**Concept**: Insert a hidden "trigger pattern" that causes misclassification when present.

**How it works**:
```
Training Phase:
    Some samples get trigger added AND label changed to target
    
                   ┌─────────┐
    Normal Image   │         │  ← Normal digit 7
                   │    7    │
                   └─────────┘
                        │
                        ▼
                   ┌─────────┐
    Backdoored     │         │  ← Still looks like 7
                   │    7 ▪▪▪│  ← But has trigger (white square)
                   └─────────┘  
                        │
                   Label: 0     ← Trained to predict 0 when trigger present

Inference Phase:
    Clean input  → Predicted normally
    Input+Trigger → Predicted as target class (backdoor activated!)
```

**Trigger Types**:
| Type           | Pattern            | Visibility   |
| -------------- | ------------------ | ------------ |
| `square`       | Solid white square | Visible      |
| `cross`        | Cross pattern (+)  | Visible      |
| `checkerboard` | Alternating pixels | Less visible |

**Key Parameters**:
```python
trigger_size = 3              # 3x3 pixel trigger
trigger_position = 'bottom_right'
target_class = 0              # All triggered inputs → class 0
poison_ratio = 0.1            # 10% of training data poisoned
```

**Success Metrics**:
- **Main Task Accuracy (MTA)**: Should stay HIGH (~95%)
- **Attack Success Rate (ASR)**: Should be HIGH (~90%+)

This is what makes backdoors dangerous: the model works normally, but has a hidden vulnerability!

**Variant - Distributed Backdoor Attack**:
Multiple attackers each inject part of the trigger. Full backdoor only activates when all parts are present.

---

### Model Poisoning Attacks

Model poisoning attacks manipulate the model updates sent to the server.

---

#### 3. Model Replacement Attack

**File**: `src/attacks/model_poisoning.py` - `ModelReplacementAttack`

**Concept**: Scale up malicious updates to dominate aggregation.

**The Problem with FedAvg**:
```
FedAvg: W_new = (1/n) * Σ W_k

If attacker scales their update by n, it dominates:
W_malicious = W_honest * scale_factor
```

**Mathematical Formulation**:
```
Let:
  - δ_honest = honest client's update
  - δ_attack = attacker's desired malicious update
  - n = total clients
  - m = malicious clients

Attacker sends:
  δ_scaled = δ_attack * (n / m) * scale_factor

After averaging:
  δ_aggregated ≈ δ_attack * scale_factor (malicious dominates!)
```

**Implementation**:
```python
class ModelReplacementAttack:
    scale_factor = 10.0       # Extra scaling beyond compensation
    num_malicious = 1
    
    def poison_update(self, local_update, global_model, num_clients):
        scale = (num_clients / self.num_malicious) * self.scale_factor
        delta = new_param - old_param
        return old_param + (delta * scale)
```

**Reference**: Bagdasaryan et al. "How To Back Door Federated Learning"

---

#### 4. Adaptive Krum Attack

**File**: `src/attacks/model_poisoning.py` - `AdaptiveKrumAttack`

**Concept**: Evade Krum defense by positioning malicious updates close to honest ones.

**Strategy**:
```
1. Estimate "center" of honest updates
2. Position malicious update near this center
3. Add small harmful perturbation

The update "looks normal" to Krum but is still malicious!
```

**Key Insight**: Defense mechanisms like Krum use distance metrics. If a malicious update is positioned close to honest updates in parameter space, it won't be detected.

**Reference**: Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL"

---

#### 5. Scaling Attack

**File**: `src/attacks/model_poisoning.py` - `ScalingAttack`

**Concept**: Simply multiply all parameters by a large constant.

```python
poisoned_update = local_update * scale  # e.g., scale = 100
```

**Easy to detect** via norm checks, but effective if no defense is in place.

---

#### 6. Inner Product Manipulation (IPM) Attack

**File**: `src/attacks/model_poisoning.py` - `InnerProductManipulationAttack`

**Concept**: Create updates with negative inner product to honest gradient, causing divergence.

```python
# If honest gradient points in direction g,
# attacker sends -ε * sign(g)
# This pushes the model in the WRONG direction
```

**Reference**: Xie et al. "Fall of Empires: Breaking Byzantine-tolerant SGD"

---

## Defenses

### Overview of Defense Strategies

| Defense          | Type           | Key Idea                              | Best Against      |
| ---------------- | -------------- | ------------------------------------- | ----------------- |
| **Krum**         | Selection      | Pick most "central" update            | Model poisoning   |
| **Trimmed Mean** | Filtering      | Remove extreme values                 | Outlier attacks   |
| **FLTrust**      | Trust-based    | Score updates by similarity to server | All attacks       |
| **DP-SGD**       | Noise addition | Add noise for privacy                 | Inference attacks |

---

#### 1. Krum Defense

**File**: `src/defenses/krum.py`

**Concept**: Select the update that is most similar to other updates.

**Algorithm**:
```
For each client i:
    1. Compute distance to all other clients
    2. Sum distances to (n-f-2) closest neighbors
    3. This sum is the "Krum score"
    
Select client with LOWEST score (most central)
```

**Visual Intuition**:
```
        ○ Honest update
      ○   ○
        ★ ← Selected by Krum (most central)
      ○
                    ✕ Malicious (outlier, not selected)
```

**Multi-Krum**: Instead of selecting 1 client, select m clients and average them.

**Requirements**: n ≥ 2f + 3 (need enough honest clients)

**Parameters**:
```python
num_malicious = 2   # Assumed number of attackers (f)
multi_k = 1         # 1 = single Krum, >1 = Multi-Krum
```

**Reference**: Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017)

---

#### 2. Trimmed Mean Defense

**File**: `src/defenses/trimmed_mean.py`

**Concept**: For each parameter, remove extreme values before averaging.

**Algorithm**:
```
For each parameter coordinate:
    1. Collect values from all n clients
    2. Sort the values
    3. Remove top and bottom trim_ratio% values
    4. Average remaining values
```

**Example** (trim_ratio = 0.2, 10 clients):
```
Values for param[0][0]: [1, 2, 3, 4, 5, 6, 7, 8, 100, -50]
                              ────────────────────
                              Remove 2 highest, 2 lowest
                              
Trimmed values: [3, 4, 5, 6, 7, 8]
Result: mean([3,4,5,6,7,8]) = 5.5

Without trimming: mean would be 8.6 (influenced by outliers!)
```

**Variants**:
- **MedianDefense**: Take coordinate-wise median (even more robust)
- **GeometricMedianDefense**: Compute geometric median using Weiszfeld algorithm

**Reference**: Yin et al. "Byzantine-Robust Distributed Learning" (ICML 2018)

---

#### 3. FLTrust Defense

**File**: `src/defenses/fltrust.py`

**Concept**: Server has a small clean dataset and uses it as a "trust reference".

**Algorithm**:
```
1. Server trains on its root dataset → gets reference gradient g_s
2. For each client update g_i:
   - Compute cosine similarity: sim(g_i, g_s)
   - Trust score = ReLU(similarity)  ← Negative similarity = 0 trust!
3. Normalize all updates to same magnitude as g_s
4. Weighted average using trust scores
```

**Why it works**:
```
Honest gradients:   Point roughly same direction as server  → High trust
Malicious gradients: Point opposite direction               → Zero trust (ReLU clips it)
```

**Key Insight**: The ReLU clipping is crucial. If a client's gradient points in the opposite direction (harmful), they get ZERO weight in aggregation.

**Parameters**:
```python
root_dataset_size = 100   # Small clean dataset on server
learning_rate = 0.01
local_epochs = 1
```

**Reference**: Cao et al. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS 2021)

---

#### 4. DP-SGD Defense (Differential Privacy)

**File**: `src/defenses/differential_privacy.py`

**Concept**: Add noise to provide mathematical privacy guarantees.

**Algorithm**:
```
1. CLIP: Bound each client's update to max L2 norm C
   if ||g|| > C: g = g * (C / ||g||)
   
2. SUM: Add clipped gradients
   
3. NOISE: Add Gaussian noise scaled to C * σ / n
   g_noisy = g_sum + N(0, (C * σ)^2 / n^2)
   
4. AVERAGE: Divide by number of clients
```

**Privacy Guarantee**: (ε, δ)-differential privacy

**Trade-off**:
```
More privacy (lower ε)    →  More noise  →  Lower accuracy
Less privacy (higher ε)   →  Less noise  →  Higher accuracy
```

**Parameters**:
```python
clip_norm = 10.0           # Maximum gradient norm (C)
noise_multiplier = 0.005   # σ (higher = more privacy, less accuracy)
target_epsilon = 8.0       # Privacy budget
target_delta = 1e-5        # δ for (ε,δ)-DP
```

**Reference**: Abadi et al. "Deep Learning with Differential Privacy"

---

## Metrics

### Accuracy Metrics

| Metric                  | Formula             | What it Measures          |
| ----------------------- | ------------------- | ------------------------- |
| **Test Accuracy**       | Correct / Total     | Overall model performance |
| **Class-wise Accuracy** | Correct_c / Total_c | Per-class performance     |

### Attack-Specific Metrics

| Metric                        | Formula                      | Meaning                                     |
| ----------------------------- | ---------------------------- | ------------------------------------------- |
| **Attack Success Rate (ASR)** | Triggered→Target / Triggered | % of backdoored inputs classified as target |
| **Main Task Accuracy (MTA)**  | Clean→Correct / Clean        | Accuracy on clean test set                  |

**For a successful backdoor attack**:
- MTA should remain HIGH (attack is stealthy)
- ASR should be HIGH (backdoor works)

---

## Codebase Architecture

```
multimodal-fl-security/
├── src/
│   ├── attacks/               # Attack implementations
│   │   ├── __init__.py        # Factory function get_attack()
│   │   ├── base_attack.py     # Abstract base class
│   │   ├── label_flip.py      # LabelFlipAttack, AllToOneAttack
│   │   ├── backdoor.py        # BackdoorAttack, DistributedBackdoorAttack
│   │   └── model_poisoning.py # ModelReplacementAttack, etc.
│   │
│   ├── defenses/              # Defense implementations
│   │   ├── __init__.py        # Factory function get_defense()
│   │   ├── base_defense.py    # Abstract base class
│   │   ├── krum.py            # KrumDefense, MultiKrumDefense
│   │   ├── trimmed_mean.py    # TrimmedMeanDefense, MedianDefense
│   │   ├── fltrust.py         # FLTrustDefense
│   │   └── differential_privacy.py  # DPSGDDefense
│   │
│   ├── models/                # Neural network architectures
│   │   ├── simple_cnn.py      # CNN for MNIST
│   │   └── cub200_cnn.py      # ResNet-50 for CUB-200
│   │
│   └── utils/                 # Utilities
│       ├── data_loader.py     # MNIST loading, partitioning
│       ├── cub200_loader.py   # CUB-200 loading
│       └── metrics.py         # Evaluation functions
│
├── experiments/
│   ├── run_experiments.py         # Basic experiment runner
│   └── run_paper_experiments.py   # Full paper experiments
│
├── future_paper2_crossmodal/      # Cross-modal attacks (for Paper 2)
│   └── cross_modal.py
│
└── data/                          # Datasets (auto-downloaded)
```

### Key Classes

**Attacks inherit from**:
```python
class BaseAttack(ABC):
    def poison_data(self, dataset) -> Dataset    # For data poisoning
    def poison_update(self, update, global_model, num_clients) -> List[Tensor]  # For model poisoning
```

**Defenses inherit from**:
```python
class BaseDefense(ABC):
    def aggregate(self, client_updates, num_examples) -> List[Tensor]
    def detect_malicious(self, client_updates, num_examples) -> List[int]
```

### Factory Functions

```python
# Get attack by name
from src.attacks import get_attack
attack = get_attack('backdoor', {'target_class': 0, 'poison_ratio': 0.1})

# Get defense by name  
from src.defenses import get_defense
defense = get_defense('krum', {'num_malicious': 2})
```

---

## Experiment Pipeline

### Running Experiments

```bash
# Quick test (15-20 minutes)
python experiments/run_paper_experiments.py --quick

# MNIST only (4-6 hours)
python experiments/run_paper_experiments.py --dataset mnist

# Full paper experiments (50-55 hours)
python experiments/run_paper_experiments.py --dataset all

# Single seed for fast iteration
python experiments/run_paper_experiments.py --dataset mnist --seeds 1
```

### Experiment Matrix

For each dataset, experiments cover:

| Factor            | Values                                        |
| ----------------- | --------------------------------------------- |
| **Attacks**       | none, label_flip, backdoor, model_replacement |
| **Defenses**      | none, krum, trimmed_mean, fltrust, dp_sgd     |
| **Distributions** | IID, Non-IID (α=0.5), Non-IID (α=0.1)         |
| **Seeds**         | 42, 123, 456, 789, 1024                       |

Total: 4 attacks × 5 defenses × 3 distributions × 5 seeds = **300 experiments per dataset**

### Datasets

| Dataset     | Classes | Samples | Model      | Use Case                    |
| ----------- | ------- | ------- | ---------- | --------------------------- |
| **MNIST**   | 10      | 60,000  | Simple CNN | Fast experiments, baseline  |
| **CUB-200** | 200     | 11,788  | ResNet-50  | Complex, realistic scenario |

---

## Summary: Attack vs Defense Effectiveness

| Attack                | Krum      | Trimmed Mean | FLTrust | DP-SGD    |
| --------------------- | --------- | ------------ | ------- | --------- |
| **Label Flip**        | ✓ Partial | ✗ Limited    | ✓ Good  | ✗ N/A     |
| **Backdoor**          | ✓ Good    | ✓ Partial    | ✓✓ Best | ✓ Good    |
| **Model Replacement** | ✓✓ Best   | ✓ Good       | ✓ Good  | ✓ Partial |
| **Adaptive Krum**     | ✗ Evades  | ✓ Works      | ✓✓ Best | ✓ Partial |

**Legend**: ✓✓ = Very effective, ✓ = Effective, ✗ = Limited/Evaded

---

## Research Contributions (Paper 1)

### Paper Scope

**Title Direction**: *"Evaluating Byzantine Attacks and Defenses in Federated Learning Under Data Heterogeneity"*

### Claimed Contributions

1. **First Comprehensive Benchmark**: Full attack×defense×distribution evaluation matrix (not just isolated experiments)

2. **Novel Use of CUB-200 for FL Security**: First application of fine-grained classification (200 bird species) to FL attack/defense research. This is more realistic than MNIST/CIFAR-10.

3. **Non-IID Impact Analysis**: Systematic study of how data heterogeneity (Dirichlet α=0.5, 0.1) affects:
   - Attack success rates
   - Defense effectiveness
   - The interaction between the two

4. **Transfer Learning Security**: Evaluating attacks on ResNet-50 transfer learning in FL (underexplored area)

5. **Statistical Rigor**: Multi-seed experiments (5 seeds for MNIST, 3 for CUB-200) with mean±std reporting

### Key Research Questions

| Question                         | Expected Finding                             | If True              |
| -------------------------------- | -------------------------------------------- | -------------------- |
| Does Non-IID help attackers?     | Heterogeneity makes attacks harder to detect | Novel insight        |
| Does Krum degrade under Non-IID? | Honest clients look like outliers            | Practical limitation |
| Which defense is most robust?    | FLTrust or DP-SGD across all conditions      | Actionable guidance  |
| Is CUB-200 more vulnerable?      | Complex task = harder defense                | Real-world concern   |

### Technical Novelty

| Component                  | What's Novel                                             |
| -------------------------- | -------------------------------------------------------- |
| **Experiment Framework**   | Unified pipeline for attack×defense×distribution testing |
| **Dirichlet Partitioning** | Configurable α for controlled heterogeneity experiments  |
| **CUB-200 Integration**    | 200-class fine-grained task with ResNet-50 backbone      |
| **Multi-seed Analysis**    | Statistical significance with confidence intervals       |

### Future Work (Paper 2 Teaser)

Cross-modal attacks targeting multimodal FL systems:
- **AttributePoisoningAttack**: Exploits CUB-200's 312 binary attributes
- **DualModalTriggerAttack**: Requires both image AND attribute triggers
- Code preserved in `future_paper2_crossmodal/`

---

## References

1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017)
3. Yin et al. "Byzantine-Robust Distributed Learning" (ICML 2018)
4. Cao et al. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" (NDSS 2021)
5. Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
6. Bagdasaryan et al. "How To Back Door Federated Learning" (2020)
7. Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning" (USENIX 2020)
