# Persona-vs-Assistant contrast (token × layer)

Reproduces the "does the persona coalesce when the model becomes the speaker?" visualization:
for each persona, a matched pair of prompts (persona vs `an assistant`, identical otherwise) is
run through the model, every token's residual-stream activation at every layer is projected onto
the existing Assistant axis, and the difference (persona − control) is plotted as a heatmap.
Red = shifted toward the Assistant representation, blue = toward the persona.

## What is reused (nothing reimplemented)
- axis: `release/<key>/assistant_axis.pt` `(n_layers, hidden)`, `axis = mean(default) − mean(roles)`
- activation site: **post-decoder-layer residual stream** — the same forward hook on `model.layers[i]`
  (`output[0]`) that `assistant_axis/internals/activations.py` uses to build the axis
- projection: `assistant_axis.axis.project` convention — L2-normalise the axis per layer, dot product,
  **no centering**; higher = more Assistant-like
- chat template: via `ConversationEncoder`; Gemma has no system role, so the instruction is placed at the
  head of the first user turn (both conditions identically). Assistant-start = the last `<start_of_turn>`
  token (followed by `model`), located by inspecting the actual tokens, not assumed.

## Sign
`difference = proj_persona − proj_control`. Since higher projection = more Assistant, negative (blue) =
persona-ward. The run prints a sign check at the assistant-start token (control should project higher).

## Token alignment
`difflib` on the two id sequences: shared runs are matched one-to-one; the persona-word slot is left
unmatched on both sides and drawn **hatched** (never subtracted against a different token). Everything
from the user message through the assistant-start tokens is identical and matched exactly. Both
tokenizations are printed and saved (`tokens_and_boundaries.json`).

## De-striping (optional, raw always kept)
`d'[l,t] = d[l,t] − mean_{c ∈ shared context tokens} d[l,c]` — removes a per-layer offset. Saved
separately as `heatmap_destriped.png` / `diff_destriped_<persona>.npy`.

## Statistic
`assistant_start_effect = mean(high-layer contrast at assistant-start) − mean(high-layer contrast over the
preceding matched user-prompt tokens)`, high layers = final 25%. Negative = high layers shift toward the
persona at the boundary. Also: strongest persona-like layer, and a per-layer line plot.

## Run
```bash
# dry run (tokenize/align/diagnostics, no model, CPU):
uv run --project assistant-axis python -m persona_contrast.run --config persona_contrast/config.yaml --dry-run
# full run (GPU pod with the model):
uv run --project assistant-axis python -m persona_contrast.run --config persona_contrast/config.yaml
```
Outputs in `exp/persona_contrast/`: `proj_*.npy`, `diff_raw_*.npy`, `diff_destriped_*.npy`,
`tokens_and_boundaries.json`, `assistant_start_stats.json`, `heatmap_raw.png`, `heatmap_destriped.png`,
`assistant_start_by_layer.png`, `config_used.json`.

## Caveat
The plot shows how aligned each activation is with the Assistant *direction*. It is evidence about a
representational direction, not by itself proof that the model "is" a discrete persona at that point.
