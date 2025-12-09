# Experiment Matrix Results

| Attack | Defense | Dataset | Partition | Clients | Accuracy | ASR |
|--------|---------|---------|-----------|---------|----------|-----|
| none | none | mnist | iid | 5 | 98.66% | - |
| none | krum | mnist | iid | 5 | 98.67% | - |
| backdoor | none | mnist | iid | 5 | 98.74% | 9.15% |
| backdoor | krum | mnist | iid | 5 | 98.68% | 0.16% |
| backdoor | trimmed_mean | mnist | iid | 5 | 98.77% | 0.22% |