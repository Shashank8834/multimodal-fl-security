# Defense Strategy Document
## Attack-Defense Compatibility Matrix

---

## 1. Defense Classification

```
                         FL DEFENSES
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   AGGREGATION           TRUST-BASED           PRIVACY
        │                     │                     │
   ┌────┴────┐           ┌────┴────┐          ┌────┴────┐
   │         │           │         │          │         │
  Krum   Trimmed       FLTrust  Reputation   DP-SGD   Clip
  Multi   Mean                                        Norm
  Krum   Median
```

---

## 2. Defense Mechanisms

### 2.1 Krum (Byzantine-Robust)

**Goal**: Select update closest to majority cluster

**Algorithm**:
```python
def krum(updates, f):
    n = len(updates)
    scores = []
    for i, u_i in enumerate(updates):
        distances = sorted([||u_i - u_j|| for j != i])
        score = sum(distances[:n-f-2])
        scores.append(score)
    return updates[argmin(scores)]
```

**Parameters**:
| Parameter | Description                |
| --------- | -------------------------- |
| f         | Assumed malicious clients  |
| k         | Multi-Krum selection count |

**Effectiveness**:
- ✅ Stops: Model replacement, scaling
- ⚠️ Partial: Backdoor (reduces ASR)
- ❌ Fails: Adaptive Krum attack

---

### 2.2 Trimmed Mean

**Goal**: Remove outliers coordinate-wise

**Algorithm**:
```python
def trimmed_mean(updates, beta):
    for each coordinate c:
        values = sorted([u[c] for u in updates])
        trim = int(len(values) * beta)
        result[c] = mean(values[trim:-trim])
    return result
```

**Parameters**:
| Parameter | Description | Range   |
| --------- | ----------- | ------- |
| beta      | Trim ratio  | 0.1-0.3 |

**Effectiveness**:
- ✅ Stops: Scaling attacks
- ✅ Partial: Byzantine attacks
- ⚠️ Weak: Coordinated attacks

---

### 2.3 Coordinate-Wise Median

**Goal**: Robust statistic per coordinate

**Algorithm**:
```python
def median(updates):
    for each coordinate c:
        result[c] = median([u[c] for u in updates])
    return result
```

**Effectiveness**: Similar to Trimmed Mean

---

### 2.4 FLTrust

**Goal**: Trust scoring with server-side validation

**Algorithm**:
```python
def fltrust(client_updates, root_data):
    server_grad = train_on_root(root_data)
    
    for u in client_updates:
        trust = max(0, cosine_sim(u, server_grad))  # ReLU
        normalized = u * ||server_grad|| / ||u||
        
    return weighted_sum(normalized, trust_scores)
```

**Parameters**:
| Parameter    | Description                     |
| ------------ | ------------------------------- |
| root_dataset | Clean server data (100 samples) |

**Effectiveness**:
- ✅ Stops: Most attacks
- ⚠️ Requires: Server-side data

---

### 2.5 DP-SGD (Differential Privacy)

**Goal**: Privacy + bounded influence

**Algorithm**:
```python
def dp_aggregate(updates, C, sigma):
    clipped = [clip(u, max_norm=C) for u in updates]
    summed = sum(clipped)
    noise = gaussian(0, C * sigma)
    return (summed + noise) / len(updates)
```

**Parameters**:
| Parameter | Description      |
| --------- | ---------------- |
| C         | Clip norm        |
| sigma     | Noise multiplier |
| epsilon   | Privacy budget   |

**Trade-off**: Privacy vs. Accuracy

---

## 3. Attack-Defense Matrix

| Attack ↓ \ Defense → | FedAvg | Krum | Trimmed | FLTrust | DP-SGD |
| -------------------- | ------ | ---- | ------- | ------- | ------ |
| Label Flip           | ❌      | ⚠️    | ⚠️       | ✅       | ⚠️      |
| Backdoor             | ❌      | ⚠️    | ⚠️       | ✅       | ⚠️      |
| Model Replacement    | ❌      | ✅    | ✅       | ✅       | ✅      |
| Scaling              | ❌      | ✅    | ✅       | ✅       | ✅      |
| Adaptive Krum        | ❌      | ❌    | ⚠️       | ✅       | ⚠️      |
| IPM                  | ❌      | ⚠️    | ✅       | ✅       | ⚠️      |
| Cross-Modal          | ❌      | ⚠️    | ⚠️       | ✅       | ⚠️      |

**Legend**: ✅ = Effective, ⚠️ = Partial, ❌ = Ineffective

---

## 4. Defense Under Non-IID Data

| Defense      | IID | Mild Non-IID | Severe Non-IID |
| ------------ | --- | ------------ | -------------- |
| Krum         | ✅   | ✅            | ⚠️              |
| Trimmed Mean | ✅   | ✅            | ⚠️              |
| Median       | ✅   | ✅            | ❌              |
| FLTrust      | ✅   | ✅            | ✅              |
| DP-SGD       | ✅   | ⚠️            | ⚠️              |

**Note**: Non-IID data increases benign client variance, making attack detection harder

---

## 5. Computational Overhead

| Defense      | Time Complexity   | Space  |
| ------------ | ----------------- | ------ |
| FedAvg       | O(n·d)            | O(d)   |
| Krum         | O(n²·d)           | O(n²)  |
| Multi-Krum   | O(n²·d)           | O(n²)  |
| Trimmed Mean | O(n·d·log n)      | O(n·d) |
| Median       | O(n·d)            | O(n·d) |
| FLTrust      | O(n·d) + training | O(d)   |
| DP-SGD       | O(n·d)            | O(d)   |

Where: n = clients, d = model parameters

---

## 6. Defense Configuration Examples

### Krum
```yaml
defense:
  type: "krum"
  num_malicious: 2
  multi_k: 1
```

### Trimmed Mean
```yaml
defense:
  type: "trimmed_mean"
  trim_ratio: 0.1
```

### FLTrust
```yaml
defense:
  type: "fltrust"
  root_dataset_size: 100
```

### DP-SGD
```yaml
defense:
  type: "dp_sgd"
  clip_norm: 1.0
  noise_multiplier: 0.1
  target_epsilon: 8.0
```

---

## 7. Implementation Summary

| Defense          | File                      | Class                     |
| ---------------- | ------------------------- | ------------------------- |
| Krum             | `krum.py`                 | `KrumDefense`             |
| Multi-Krum       | `krum.py`                 | `MultiKrumDefense`        |
| Trimmed Mean     | `trimmed_mean.py`         | `TrimmedMeanDefense`      |
| Median           | `trimmed_mean.py`         | `MedianDefense`           |
| Geometric Median | `trimmed_mean.py`         | `GeometricMedianDefense`  |
| FLTrust          | `fltrust.py`              | `FLTrustDefense`          |
| DP-SGD           | `differential_privacy.py` | `DPSGDDefense`            |
| Gradient Clip    | `differential_privacy.py` | `GradientClippingDefense` |
| Norm Bound       | `differential_privacy.py` | `NormBoundingDefense`     |

---

## 8. Recommendations

1. **Default**: Use Krum for Byzantine robustness
2. **High Security**: Use FLTrust with clean root data
3. **Privacy Required**: Use DP-SGD with tuned epsilon
4. **Non-IID Data**: Prefer FLTrust over distance-based methods

---

## 9. References

1. Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" NeurIPS 2017
2. Yin et al. "Byzantine-Robust Distributed Learning" ICML 2018
3. Cao et al. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" NDSS 2021
4. Abadi et al. "Deep Learning with Differential Privacy" CCS 2016
