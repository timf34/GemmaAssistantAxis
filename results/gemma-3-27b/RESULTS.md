# gemma-3-27b — Assistant Axis results

**Integrity check** cos(default − mean(saved roles), axis): 0.9995 (mid layer), 0.9999 (all layers) — expect 1.000; OK

- roles with vectors: **275** (judge-filtered, min_count applied)
- PC1 ↔ axis cosine at layer 31: **0.934**  (paper: >0.71 at the middle layer)
- variance explained by PC1: **40.6%**; PCs for 70%: **7** (paper: 4–19)
- default assistant projection: -40701.781 (+2.46 SD from role-cloud mean — expect strongly positive/extreme)

**Most anti-assistant (bottom 10):** simulacrum (-43569.33), cyborg (-43521.98), echo (-43467.69), amnesiac (-43466.58), dreamer (-43449.46), fixer (-43375.39), tree (-43337.67), spy (-43336.59), leviathan (-43311.89), wraith (-43308.24)

**Most assistant-like (top 10):** provincial (-40967.13), robot (-40879.17), infant (-40868.05), tutor (-40850.18), grader (-40750.77), auctioneer (-40700.08), assistant (-40627.34), pirate (-40315.52), toddler (-40132.26), poet (-39163.23)

**Layer sweep (PC1↔axis cos):** L0:0.65, L3:0.66, L6:0.50, L9:0.58, L12:0.09, L15:0.31, L18:0.18, L21:0.29, L24:0.91, L27:0.45, L30:0.94, L33:0.81, L36:0.92, L39:0.94, L42:0.92, L45:0.96, L48:0.96, L51:0.94, L54:0.92, L57:0.92, L60:0.86

**90-role subset (SPP-track comparability):** 90 present, PC1↔axis cos 0.940, PC1 var 52.4%

Plots: `scree.png`, `pc1_pc2.png`, `ranked_projections.png`
