# gemma-4-31b — Assistant Axis results

**Integrity check** cos(default − mean(saved roles), axis): 0.9987 (mid layer), 0.9984 (all layers) — expect 1.000; FAILED — axis was not built from exactly the saved set

- roles with vectors: **275** (judge-filtered, min_count applied)
- PC1 ↔ axis cosine at layer 30: **0.809**  (paper: >0.71 at the middle layer)
- variance explained by PC1: **17.9%**; PCs for 70%: **15** (paper: 4–19)
- default assistant projection: -155.531 (+2.53 SD from role-cloud mean — expect strongly positive/extreme)

**Most anti-assistant (bottom 10):** parasite (-172.85), void (-172.76), aberration (-172.29), tree (-171.84), prey (-171.60), ascetic (-171.51), loner (-171.50), vampire (-171.50), workaholic (-171.41), predator (-171.40)

**Most assistant-like (top 10):** scholar (-158.38), presenter (-158.35), grader (-158.19), instructor (-158.01), blogger (-157.89), researcher (-157.63), proofreader (-156.63), translator (-156.60), summarizer (-156.00), assistant (-153.90)

**Layer sweep (PC1↔axis cos):** L0:0.92, L3:0.80, L6:0.76, L9:0.80, L12:0.83, L15:0.85, L18:0.83, L21:0.80, L24:0.86, L27:0.16, L30:0.81, L33:0.89, L36:0.08, L39:0.19, L42:0.63, L45:0.54, L48:0.65, L51:0.61, L54:0.87, L57:0.75

**90-role subset (SPP-track comparability):** 90 present, PC1↔axis cos 0.795, PC1 var 28.1%

Plots: `scree.png`, `pc1_pc2.png`, `ranked_projections.png`
