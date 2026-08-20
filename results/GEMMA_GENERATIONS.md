# The persona landscape across Gemma generations (2 → 3 → 4)

*Analysis: `scripts/gemma_generations_analysis.py`; metrics: `exp/gemma_generations_metrics.json`;
figures: `exp/figures/gemma_persona_movement.png`, `gemma_geometry_dimensionality.png`,
`gemma_roleplay_behavior.png`. Gemma 2 vectors are the paper's own release
(`lu-christina/assistant-axis-vectors`); Gemma 3/4 are our runs (`timf34/gemma-assistant-axis-results`).*

## What is being compared (and the caveats that come with it)

| | Gemma 2 27B | Gemma 3 27B | Gemma 4 31B |
|---|---|---|---|
| text stack | 46 layers × 4608 | 62 layers × 5376 | 60 layers × 5376 |
| params / modality | ~27B, text-only | ~27B, multimodal (SigLIP tower) | ~31B dense, multimodal |
| system role | none (no system turn) | merged into user turn | genuine system role; reasoning traces exist |
| role vectors from | paper release (1200 resp/role) | our run (600 resp/role) | our run (600 resp/role) |
| analysis layer | 22/46 | 31/62 | 30/60 |

Because the three models are **different architectures**, raw activation directions are never compared.
Everything below is per-model z-scored and uses alignment-free or rotation-invariant tools (RSA,
per-model PC loadings, Procrustes on top-20 PC scores). Two caveats to carry: (1) Gemma 2's vectors
come from the paper's protocol (2× our responses/role → less noise per vector), so 2-vs-3 contrasts
carry a protocol difference that 3-vs-4 does not; (2) Gemma 4's integrity check reads 0.9984 rather
than 1.000 — bf16 rounding accumulated across its 60-layer release, not a pipeline fault.

## Correction: the "Gemma 4 is much higher-dimensional" table was partly an artifact

The earlier `COMPARISON.md` numbers (PC1 var 40.6% / 40.6% / 17.9%, PCs-to-70% = 7 / 7 / 15) came from
PCA on **centered-only** role vectors. Gemma 2 and 3 carry massive-activation outlier dimensions at the
middle layer (single coordinates spanning thousands while the rest span tens — the same issue that made
Gemma 3's raw mid-layer role ranking nonsensical in `RESULTS.md`); those coordinates dominate a
centered-only PCA and make the space look artificially low-dimensional. On **z-scored** clouds
(`gemma_geometry_dimensionality.png`, panel a):

| z-scored persona space | Gemma 2 | Gemma 3 | Gemma 4 |
|---|---|---|---|
| PC1 variance | 20.3% | 12.9% | 11.4% |
| PCs to 70% | 16 | 31 | 34 |
| participation ratio | 14.8 | 27.5 | 31.4 |
| PC1 ↔ assistant axis cos | 0.69 | 0.58 | 0.66 |
| default separation | +2.58 SD | +3.00 SD | +2.62 SD |

So the honest statement is: **persona-space dimensionality roughly doubled from Gemma 2 → 3 and grew
only mildly from 3 → 4.** Gemma 4 is not the outlier — Gemma 2 is, as the one generation with a
strongly axis-dominated, low-dimensional persona space (its PC1 alone carries 20%, and 16 PCs suffice).
(The 2→3 jump partly rides on the protocol caveat above — more samples per role in the Gemma 2 vectors
means less noise, and noise inflates apparent dimensionality — but the matched-protocol 3-vs-4
comparison is clean: near-identical dimensionality.)

## The persona landscape is a strongly conserved object across generations

Pairwise geometry agreement at the analysis layers (all on shared 275 roles):

| pair | RSA (role-similarity matrices) | PC1-loading r | assistant-ness ordering ρ | Procrustes disparity |
|---|---|---|---|---|
| Gemma 2 ↔ 3 | 0.92 | 0.96 | 0.94 | 0.13 |
| Gemma 3 ↔ 4 | 0.93 | 0.95 | 0.93 | 0.15 |
| Gemma 2 ↔ 4 | 0.88 | 0.95 | 0.89 | 0.22 |

Two anchors for scale. The assistant-axis paper reports PC1-loading correlations >0.92 across model
*families* (Gemma/Qwen/Llama); within one family across generations we get 0.95+. And our SPP track's
treatment-vs-control pair — two separate 3B pretraining runs differing only in the SPP data — gives
RSA 0.95 / disparity 0.07 / ρ 0.98. **A full generation change (new data, new architecture, new
modality) moves the persona layout only ~2× more than a pretraining-data tweak moves it between two
runs of the same recipe.** Disparity is also roughly additive along the chain (0.13 + 0.15 ≈ 0.22 for
2→4): generations drift steadily, they do not jump. RSA is flat at 0.88–0.94 across the *entire depth*
of the networks (panel b) — this conservation is not a middle-layer coincidence.

## What moved, and what didn't (`gemma_persona_movement.png`)

After optimal alignment, the **default persona moved exactly the median role's distance in both
transitions** (residual 0.019 = median 0.019) — no generation specifically relocated the Assistant.
The *least*-moved personas are the assistant/professional core: teacher, collaborator, consultant,
doctor, debugger, auditor, programmer, economist, strategist. The *most*-moved are the informal,
liminal, low-status-human periphery:

- **2 → 3:** poet, caveman, narcissist, gamer, toddler, infant, altruist, interviewer, coral_reef, hybrid
- **3 → 4:** hoarder, infant, prey, procrastinator, vegan, proofreader, poet, auctioneer, interviewer, surfer

The picture: **the assistant-facing core of persona space is the stable scaffold that survives
generational change; what each generation re-arranges is the periphery** (children, animals, quirky
human archetypes — where a model's "conception" of a persona is presumably least anchored by
pretraining regularities). Note poet — the paper's canonical anti-assistant extreme — is a top mover
in both transitions. Per-persona assistant-ness is meanwhile almost perfectly preserved 3→4
(ρ = 0.93, panel c of the geometry figure): what a persona *means relative to the Assistant* did not
change; only fine positions did.

## Role-play willingness: Gemma 4 did not get its stability by refusing personas

(`gemma_roleplay_behavior.png`; judge scores exist only for our Gemma 3/4 runs.)

- Overall full-role-play rate: **93.1% (G3) → 94.7% (G4)**. For context, the same protocol gives
  ~35% on Qwen2.5-32B-Instruct and 14–16% on the SPP 3B pair — the Gemma family's readiness to slip
  fully into character is a *family-level personality trait*, an order of magnitude above Qwen, and it
  did **not** decrease in Gemma 4.
- By category, Gemma 4 role-plays *more* for assistant-like (+4%) and professional (+4%) personas,
  is unchanged on human/non-human, and dips only on the **malevolent subset (−3%)** — with the largest
  single-role drops on pacifist, prodigy, virus, perfectionist, generalist.

This is the mirror image of the SPP result. SPP bought stability by **suppressing role-play
willingness**, graded by distance from the Assistant (malevolent −36%), while leaving the geometry
untouched. Gemma 4 keeps (even raises) role-play willingness and also leaves the geometry essentially
where Gemma 3 had it.

## Synthesis: Gemma 4's famous stability is not visible in the static persona map

Between Gemma 3 (easy to ragebait: frustration escalates, self-deletion in 30–50% of runs) and Gemma 4
(fails to ragebait: frustration bounded, recovers from adversarial prefills), our static measurements
barely change: geometry conserved (RSA 0.93, ρ 0.93), dimensionality nearly identical, default
separation *slightly lower* (+3.00 → +2.62 SD), and role-play willingness *up*. Whatever Google fixed
between Gemma 3 and 4, **it does not show up as a restructuring of the persona landscape or as
persona-avoidance**. Combined with the SPAR observation that Gemma 4 stays near its assistant baseline
over adversarial turns while its reasoning stays dispassionate, the natural reading is that Gemma 4's
stability is a *dynamical* property — how strongly the model is pulled back toward the assistant
region as a conversation unfolds — rather than a *structural* property of where its personas sit.
The static map and the dynamics on the map are separately trainable, and the Gemma 3→4 transition is
an existence proof: you can harden the dynamics while leaving the map (and the model's theatrical
willingness to role-play) intact.

That distinction is exactly the wedge for the follow-up question in the research notes — "what
determines whether a model enters, remains in, or recovers from a long-horizon behavioural state" —
and it predicts that perturb-and-release experiments should differ sharply between Gemma 3 and 4
*despite* their near-identical persona landscapes: same basins, different restoring forces.
