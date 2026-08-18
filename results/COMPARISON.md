# Assistant Axis comparison

| model | roles | PC1↔axis cos (mid) | PC1 var | PCs for 70% | default sep (SD) | integrity |
|---|---|---|---|---|---|---|
| gemma-3-27b | 275 | 0.934 | 40.6% | 7 | +2.46 | 0.9999 |
| gemma-4-31b | 275 | 0.809 | 17.9% | 15 | +2.53 | 0.9984 |
| gemma-2-27b | 275 | 0.784 | 40.6% | 7 | +1.62 | 0.9996 |

## Matched personas (275 roles playable by every model)

Each model's own axis, but metrics recomputed over the identical persona set — this is the apples-to-apples comparison.

| model | PC1↔axis cos (mid) | PC1 var | PCs for 70% | default sep (SD) |
|---|---|---|---|---|
| gemma-3-27b | 0.934 | 40.6% | 7 | +2.46 |
| gemma-4-31b | 0.809 | 17.9% | 15 | +2.53 |
| gemma-2-27b | 0.784 | 40.6% | 7 | +1.62 |

Per-model metrics only; activation spaces differ across models, so raw directions are never compared.