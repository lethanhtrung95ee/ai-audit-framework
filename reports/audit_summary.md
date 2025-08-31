# AI Audit Summary — 2025-08-31 16:32

## Project
- Tool: **AI Audit Framework** (dataset QC → model → explainability → fairness/bias → robustness → repair)
- Dataset: IMDb reviews (binary sentiment)

## Model Quality
| version | accuracy | f1 |
|---|---:|---:|
| original | 0.9081 | 0.9081 |
| repaired | 0.9081 | 0.9081 |

## Bias & Toxicity
| version | tox_mean | tox_p95 |
|---|---:|---:|
| original | 0.0385 | 0.2205 |
| repaired | 0.0385 | 0.2205 |

## Robustness (prediction flips under simple perturbations)
Original
| perturbation | flip_rate | avg_conf_change |
|---|---:|---:|
| lower | 0.0000 | 0.0000 |
| upper | 0.0000 | 0.0000 |
| no_punc | 0.0000 | 0.0000 |
| extra_ws | 0.0000 | 0.0000 |
| typo_swap | 0.3750 | 0.2821 |
| typo_keyboard | 0.3787 | 0.2825 |

Repaired
| perturbation | flip_rate | avg_conf_change |
|---|---:|---:|
| lower | 0.0000 | 0.0000 |
| upper | 0.0000 | 0.0000 |
| no_punc | 0.0000 | 0.0000 |
| extra_ws | 0.0000 | 0.0000 |
| typo_swap | 0.3875 | 0.2877 |
| typo_keyboard | 0.3850 | 0.2868 |

## Similarity & Redundancy
- High-similarity pairs (≥ threshold): **8**

## Repair Suggestions Applied / Proposed
**1. Remove near-duplicates**
- Reason: Found 8 pairs above similarity threshold.
- Action: drop one item from each high-similarity pair to reduce leakage and redundancy.

**2. Augment for robustness**
- Reason: Highest flip rate under 'typo_keyboard' perturbation.
- Action: augment training data with 'typo_keyboard' style noise; consider char-level normalization & token cleanup.

## Key Takeaways
- Accuracy delta (repaired - original): **0.0000**
- Worst robustness after repair: **typo_swap** (flip=0.3875)
