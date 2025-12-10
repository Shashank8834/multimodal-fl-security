# Data Heterogeneity Documentation
## Multimodal FL Security Framework

---

## 1. Overview

Data heterogeneity (Non-IID) is a key challenge in Federated Learning. This document describes how we partition data and quantify heterogeneity.

---

## 2. Partition Strategies

### 2.1 IID Partition (Baseline)

**Method**: Random uniform split

```python
indices = shuffle(range(len(dataset)))
client_indices[i] = indices[i * chunk_size : (i+1) * chunk_size]
```

**Properties**:
- Each client has equal data
- Class distribution matches global distribution
- Heterogeneity score ≈ 0

---

### 2.2 Non-IID Partition (Dirichlet)

**Method**: Dirichlet distribution for class allocation

```python
for each class c:
    proportions = Dirichlet([α, α, ..., α])  # α controls heterogeneity
    allocate class c samples according to proportions
```

**Alpha (α) Parameter**:
| α Value | Heterogeneity Level                 |
| ------- | ----------------------------------- |
| 100+    | Near-IID                            |
| 1.0     | Moderate                            |
| 0.5     | High                                |
| 0.1     | Extreme (some clients miss classes) |

**Our default**: α = 0.5 (realistic heterogeneity)

---

## 3. Heterogeneity Metrics

### 3.1 Earth Mover Distance (EMD)

Measures distance between client and global distribution:

```
EMD_i = Σ |p_i(c) - p_global(c)| / 2
```

- **Range**: 0 (IID) to 1 (completely different)
- **Threshold**: EMD > 0.3 indicates significant heterogeneity

### 3.2 Class Coverage

Fraction of total classes present at each client:

```
Coverage_i = |classes_i| / |all_classes|
```

- **IID**: ~100% coverage
- **Non-IID**: May be 30-70%

### 3.3 Label Distribution Variance

Variance of class proportions across clients:

```
Var = mean(variance(p(c)) for c in classes)
```

- Higher variance = more heterogeneous

### 3.4 Combined Heterogeneity Score

```
Score = avg_EMD × (1 - avg_coverage)
```

- **Range**: 0 (IID) to ~0.5 (extreme)
- Captures both distribution shift and missing classes

---

## 4. Dataset Statistics

### MNIST
| Stat              | Value   |
| ----------------- | ------- |
| Total samples     | 60,000  |
| Classes           | 10      |
| Image shape       | 1×28×28 |
| Mean (normalized) | 0.1307  |
| Std (normalized)  | 0.3081  |

### CUB-200
| Stat            | Value                 |
| --------------- | --------------------- |
| Total samples   | 5,994 (train)         |
| Classes         | 200                   |
| Image shape     | 3×224×224             |
| Mean (ImageNet) | [0.485, 0.456, 0.406] |
| Std (ImageNet)  | [0.229, 0.224, 0.225] |

---

## 5. Partition Examples

### 5 Clients, IID
```
Client 0: 12000 samples, 10 classes (100% coverage)
Client 1: 12000 samples, 10 classes (100% coverage)
Client 2: 12000 samples, 10 classes (100% coverage)
Client 3: 12000 samples, 10 classes (100% coverage)
Client 4: 12000 samples, 10 classes (100% coverage)

Heterogeneity Score: 0.00
```

### 5 Clients, Dirichlet (α=0.5)
```
Client 0: 11500 samples, 8 classes (80% coverage)
Client 1: 12800 samples, 9 classes (90% coverage)
Client 2: 10200 samples, 7 classes (70% coverage)
Client 3: 13000 samples, 10 classes (100% coverage)
Client 4: 12500 samples, 8 classes (80% coverage)

Heterogeneity Score: 0.15
```

---

## 6. Impact on Attacks and Defenses

| Scenario       | Attack Effectiveness  | Defense Difficulty              |
| -------------- | --------------------- | ------------------------------- |
| IID            | Baseline              | Easier (clear outliers)         |
| Mild Non-IID   | Similar               | Moderate                        |
| Severe Non-IID | May increase/decrease | Harder (benign looks malicious) |

**Key Insight**: Non-IID makes Byzantine defenses harder because honest clients have diverse updates that may appear malicious.

---

## 7. Running EDA

```bash
python experiments/eda_analysis.py
```

Output:
- Class distribution plots
- Client partition heatmaps
- Heterogeneity metrics
