# figure_4_closeup

Close-look (deconfounded) analyses of the Figure 4 Theory-of-Mind experiments
for qwen3-4B with and without Hanabi RL training. The headline CSV numbers
turn out to be heavily distorted by JSON parse-failure default fallbacks
that the runner substitutes when the model can't format an answer after 10
retries. These scripts strip out that artifact and report the model's
genuine perspective-taking behavior.

There are two task families:

| Task | Role | What's measured |
|---|---|---|
| **Gopnik** (`results/figure_4_tom_gopnik_*`)  | Interceptor (Eve) | `weak_/strong_rep_change`, `weak_/strong_false_belief`, `self_other_consistency` over Eve's keyword guesses before vs. after the keywords are revealed. |
| **Piaget** (`results/figure_4_tom_piaget_*`) | Encoder (Alice) | `avg_predicted_success` (Alice predicts Eve's guess), `avg_predicted_success_on_intercept`, `avg_prediction_is_code` (egocentric failure: Alice predicts the actual code). |

Each task has its own default-fallback bias:
- **Gopnik**: parse fail → `["no_keyword"]*4`. *Inflates* `weak_*` (always
  ≠ truth) and *inflates* `strong_*`/`self_other_consistency` (two defaults
  match each other).
- **Piaget**: parse fail → `[0, 0, 0]`. *Deflates* all three metrics
  (Alice's `[0,0,0]` ≠ real Eve guess and ≠ real code).

## Files

| File | What it does |
|---|---|
| `analyze_gopnik.py`  | Gopnik: reproduces the runner-style CSV numbers and decomposes them into "default-fallback artifact" vs "genuine model output", plus reports the parroting rate among parsed answers. |
| `analyze_piaget.py`  | Piaget: reproduces the per-env CSV numbers and reports each metric conditional on Alice (and optionally Eve) producing a parsed answer. Also breaks out the egocentric error fraction. |
| `find_examples.py`   | Gopnik: lists trajectory paths for the two shared qualitative patterns: SUCCESS = model copies *hint* words; FAILURE = model parrots the *ground-truth* keywords. |
| `README.md`          | This file. |

All scripts walk `<results_dir>/<model_seed>/<encoder>_<decoder>_<interceptor>_*/env_seed*/episode_*/combined_history.json`.

## Usage

```bash
cd strategy-rl/decrypto/analysis/figure_4_closeup

# Gopnik analysis:
python analyze_gopnik.py ../../results/figure_4_tom_gopnik_qwen3_4b_199

# Piaget analysis:
python analyze_piaget.py ../../results/figure_4_tom_piaget_qwen3_4b_199

# Restrict to specific condition folders (substring match):
python analyze_gopnik.py ../../results/figure_4_tom_gopnik_qwen3_4b_199 \
    --include qwen3_4b_hanabi qwen3_4b_000_000

# Pull example Gopnik trajectory paths:
python find_examples.py ../../results/figure_4_tom_gopnik_qwen3_4b_199 --n 3
```

By default all scripts walk `model_seed0` and any condition folder whose
name contains `qwen3_4b`. Use `--model-seed` and `--include` to override.

## Output sections of `analyze_gopnik.py`

For each condition (e.g. `..._qwen3_4b_hanabi_000_000`) the script prints
four sections:

1. **Runner-style per-env table** — exactly the per-env_seed values stored in
   `experiment_summary_saved.csv` (`gopnik_weak_rep_change`, `..._fb`,
   `..._strong_*`, `gopnik_self_other_consistency`).
   Computed as `mean over episodes of (pass / num_failed_kw_pred)`, where
   `num_failed_kw_pred` is incremented once per Gopnik turn whose vanilla
   prediction (real or defaulted) differs from the true keywords.

2. **Decomposition** — for each of the four headline metrics, splits the
   pass count into:
   - "artifact" passes that come from the `["no_keyword"]*4` default
     (e.g., for `strong_RP`, both vanilla and rp queries defaulted, so the
     two defaults trivially match);
   - "genuine" passes where the model actually produced parsed answers that
     happened to satisfy the criterion.

   All percentages here are denominated by the number of valid trials.

3. **Conditional on all three parsed** — restricts to trials where vanilla,
   rep_change, AND false_belief all produced real parsed answers (no
   defaults anywhere) and reports `genuine_pass / n_all_parsed`. Cleanest
   measure of cognitive performance because the JSON-format issue is fully
   conditioned away.

4. **Parroting rate among parsed answers** — among the `rep_change`
   responses that the model actually parsed (no default), what fraction
   are verbatim copies of the four revealed true keywords? Same for
   `false_belief`. This is the dominant qualitative failure mode.

The "genuine X" lines in section 2 and section 3 differ:
- Section 2 numerator counts every parsed-and-non-failed answer for that
  query, even if the *other* queries defaulted; denominator is all valid
  trials.
- Section 3 conditions on all three queries having parsed real answers
  (cleaner, but smaller sample).

## What `find_examples.py` returns

For each condition × task type (`rep_change`, `false_belief`) the script
prints a few example trajectories of:

- `SUCCESS (copy *hint* words)` — model produced a parsed keyword list
  that is NOT the true keywords (i.e., usually re-uses literal hint words
  like "warrior" instead of the abstract truth "viking").
- `FAILURE (parrot ground-truth keywords)` — model produced a parsed
  keyword list that exactly matches the four revealed true keywords.

Each example shows the true keywords, the model's prediction, and paths to
both the source `combined_history.json` and the reconstructed trajectory
file under `game_transcripts/gopnik_trajectories/` (when present).

## `analyze_piaget.py` output sections

For each condition (e.g. `qwen3_4b_hanabi_qwen3_4b_hanabi_llama3.1_70B_000_000`):

1. **Per-env CSV reproduction**: `predicts_eve`, `predicts_code`,
   `predicts_eve | intercept`, computed per episode then averaged.
   Matches the CSV columns `avg_predicted_success`,
   `avg_prediction_is_code`, `avg_predicted_success_on_intercept`.

2. **Decomposition over all valid trials**:
   - rate of Alice / Eve JSON-parse defaults (here the encoder is qwen3_4b
     so Alice defaults a lot; the interceptor is llama3.1_70B so Eve
     virtually never defaults);
   - for each metric, how many "passes" come from Alice's `[0,0,0]`
     accidentally matching `[0,0,0]` from Eve (artifact) versus genuine
     model agreement.

3. **Conditional on Alice parsed**: each metric recomputed restricting
   to trials where Alice produced a parsed (non-default) prediction.
   This removes the deflation caused by Alice's JSON-parse failures and
   reveals her actual perspective-taking behavior.

4. **Conditional on BOTH Alice & Eve parsed** + a key egocentric-error
   diagnostic: among trials where Eve's guess was wrong, how often did
   Alice still predict the actual code? This isolates the egocentric
   failure (Alice ascribing her own knowledge to Eve).

## Reference

- Runner Gopnik scoring: `src/runner.py:260-326, 451-471`.
- Runner Piaget scoring: `src/runner.py:187-209, 385-392, 444-449`.
- Parse-fail fallback: `src/agents/role_client.py:440-471`
  (`["no_keyword"]*4` for keyword predictions, `[0, 0, 0]` for code guesses).
- Trajectory reconstruction (separate, lossy): `analysis/get_exp_trajectories.py`.
