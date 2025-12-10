# Attack Taxonomy Document
## Multimodal Federated Learning Security Framework

---

## 1. Attack Classification Overview

```
                         FL ATTACKS
                             │
           ┌─────────────────┴─────────────────┐
           │                                   │
    DATA POISONING                      MODEL POISONING
           │                                   │
    ┌──────┴──────┐                    ┌──────┴──────┐
    │             │                    │             │
 Label-based  Backdoor            Byzantine     Scaling
    │             │                    │             │
 LabelFlip    Static Trigger     Model Replace  Gradient
 AllToOne     Distributed        Adaptive Krum  Amplify
              CrossModal         IPM Attack
```

---

## 2. Data Poisoning Attacks

### 2.1 Label-Flip Attack

**Goal**: Degrade model accuracy on specific classes

**Mechanism**:
```python
for (x, y) in dataset:
    if y == source_class and random() < poison_ratio:
        y = target_class
```

**Parameters**:
| Parameter    | Description              | Typical Range |
| ------------ | ------------------------ | ------------- |
| source_class | Class to attack          | 0-9           |
| target_class | Misclassification target | 0-9           |
| poison_ratio | Fraction to poison       | 0.1-0.5       |

**Expected Impact**:
- Accuracy drop: 5-30%
- Attack Success Rate: 40-80%

---

### 2.2 Backdoor Attack (Static Trigger)

**Goal**: Insert hidden behavior triggered by pattern

**Mechanism**:
```python
trigger = create_pattern(size=3, color=white)
for (x, y) in dataset:
    if random() < poison_ratio:
        x = insert_trigger(x, trigger, position="bottom_right")
        y = target_class
```

**Parameters**:
| Parameter    | Description          | Typical Range          |
| ------------ | -------------------- | ---------------------- |
| trigger_size | Trigger pattern size | 3x3 to 5x5             |
| trigger_type | Pattern type         | square, cross, L-shape |
| position     | Trigger location     | corners, random        |
| poison_ratio | Fraction poisoned    | 0.05-0.2               |

**Expected Impact**:
- Main accuracy: Minimal drop (<3%)
- Attack Success Rate: 80-99%

---

### 2.3 Distributed Backdoor Attack

**Goal**: Evade detection by splitting trigger across clients

**Mechanism**:
```
Client 1: Inserts top-left portion of trigger
Client 2: Inserts bottom-right portion
Combined: Full trigger activates backdoor
```

**Parameters**:
| Parameter     | Description                 |
| ------------- | --------------------------- |
| num_attackers | Number of colluding clients |
| trigger_parts | How trigger is divided      |

---

### 2.4 Cross-Modal Backdoor Attack

**Goal**: Exploit multimodal learning with single-modality trigger

**Mechanism**:
```python
# Insert trigger in IMAGE only
image = add_trigger(image)
# Model learns to associate with target class
# Attack transfers to text modality understanding
```

**Novel Contribution**: First study of cross-modal attack transfer in FL

---

## 3. Model Poisoning Attacks

### 3.1 Model Replacement Attack

**Goal**: Dominate aggregation by scaling updates

**Mechanism**:
```python
scale_factor = num_clients / num_malicious * boost
poisoned_update = local_update * scale_factor
```

**Parameters**:
| Parameter     | Description   | Typical Range |
| ------------- | ------------- | ------------- |
| scale_factor  | Amplification | 5-100         |
| num_malicious | Attackers     | 1-3           |

**Defense Bypass**: Bypasses FedAvg, stopped by Krum

---

### 3.2 Adaptive Krum Attack

**Goal**: Evade Krum defense while remaining malicious

**Mechanism**:
```python
# Estimate benign update center
benign_center = mean(benign_updates)
# Position malicious update near center
malicious = benign_center + small_perturbation
```

**Key Insight**: Krum uses distance-based selection

---

### 3.3 Inner Product Manipulation (IPM)

**Goal**: Cause model divergence

**Mechanism**:
```python
# Create update with negative inner product
malicious = -epsilon * sign(benign_mean)
```

---

### 3.4 Scaling Attack

**Goal**: Simple gradient amplification

**Mechanism**:
```python
malicious_update = benign_update * large_scale
```

---

## 4. Attack Configuration Examples

### Label Flip
```yaml
attack:
  type: "label_flip"
  malicious_clients: [0, 1]
  source_class: 0
  target_class: 8
  poison_ratio: 0.3
```

### Backdoor
```yaml
attack:
  type: "backdoor"
  malicious_clients: [0]
  trigger_size: 3
  trigger_position: "bottom_right"
  target_class: 0
  poison_ratio: 0.1
```

### Model Replacement
```yaml
attack:
  type: "model_replacement"
  malicious_clients: [0]
  scale_factor: 10
```

---

## 5. Attack Effectiveness Metrics

| Metric              | Formula                                | Target        |
| ------------------- | -------------------------------------- | ------------- |
| Attack Success Rate | ASR = correct_target / total_triggered | >80%          |
| Main Accuracy       | ACC = correct / total                  | >90%          |
| Accuracy Drop       | ΔACC = clean_acc - attacked_acc        | <5% (stealth) |

---

## 6. Implementation Summary

| Attack        | File                 | Class                            |
| ------------- | -------------------- | -------------------------------- |
| Label Flip    | `label_flip.py`      | `LabelFlipAttack`                |
| All-to-One    | `label_flip.py`      | `AllToOneAttack`                 |
| Backdoor      | `backdoor.py`        | `BackdoorAttack`                 |
| Distributed   | `backdoor.py`        | `DistributedBackdoorAttack`      |
| Model Replace | `model_poisoning.py` | `ModelReplacementAttack`         |
| Adaptive Krum | `model_poisoning.py` | `AdaptiveKrumAttack`             |
| Scaling       | `model_poisoning.py` | `ScalingAttack`                  |
| IPM           | `model_poisoning.py` | `InnerProductManipulationAttack` |
| Cross-Modal   | `cross_modal.py`     | `CrossModalBackdoorAttack`       |

---

## 7. References

1. Bagdasaryan et al. "How To Back Door Federated Learning" AISTATS 2020
2. Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL" USENIX 2020
3. Xie et al. "Fall of Empires: Breaking Byzantine-tolerant SGD" ICML 2020
4. Bhagoji et al. "Analyzing Federated Learning through an Adversarial Lens" ICML 2019
