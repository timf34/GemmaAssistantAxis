2026-08-17T20:23:43Z  doctor: driver CUDA 12.4 < 12.8 — pod unusable for this stack
2026-08-17T20:27:19Z  doctor: driver CUDA 12.4, no compat package
2026-08-17T22:28:33Z  doctor: 1 check(s) failed
2026-08-17T22:48:38Z  doctor: all checks passed
2026-08-17T22:51:49Z  doctor: all checks passed
2026-08-17T22:55:33Z  doctor: all checks passed
2026-08-17T23:01:11Z  doctor: all checks passed
2026-08-17T23:20:45Z  doctor: all checks passed
2026-08-17T23:39:26Z  doctor: all checks passed
2026-08-17T23:55:29Z  doctor: all checks passed
2026-08-18T00:08:11Z  preflight OK: google/gemma-4-31B-it (GPU 0)
2026-08-18T00:08:11Z  [gemma-4-31b] supervisor attempt 1/8
2026-08-18T00:08:11Z  [gemma-4-31b] generate: start
2026-08-18T07:13:24Z  [gemma-4-31b] generate: done
2026-08-18T07:13:24Z  [gemma-4-31b] activations: start
2026-08-18T07:13:24Z  [gemma-4-31b] judge: start
2026-08-18T08:47:29Z  [gemma-4-31b] judge: done
2026-08-18T11:47:39Z  [gemma-4-31b] activations: done
2026-08-18T11:47:39Z  [gemma-4-31b] judge_rerun: start
2026-08-18T11:48:54Z  [gemma-4-31b] judge_rerun: done
2026-08-18T11:48:54Z  [gemma-4-31b] vectors: start
2026-08-18T11:59:41Z  [gemma-4-31b] vectors: done
2026-08-18T11:59:41Z  [gemma-4-31b] axis: start
2026-08-18T12:00:12Z  [gemma-4-31b] axis: done
2026-08-18T12:00:12Z  [gemma-4-31b] package: start
2026-08-18T12:00:38Z  [gemma-4-31b] package: done
2026-08-18T12:00:38Z  [gemma-4-31b] analyze: start
2026-08-18T12:01:38Z  [gemma-4-31b] analyze: done
2026-08-18T12:02:21Z  [gemma-4-31b] upload: done (attempt 1)
2026-08-18T12:02:21Z  [gemma-4-31b] activations upload: start (100G)
2026-08-18T12:02:31Z  [gemma-4-31b] !! activations upload FAILED — kept on disk at /workspace/exp/gemma-4-31b/activations
2026-08-18T12:02:31Z  [gemma-4-31b] COMPLETE
2026-08-18T12:02:31Z  [gemma-4-31b] supervisor: success on attempt 1
