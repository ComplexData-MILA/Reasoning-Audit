# Reasoning Audit

An experimental audit of how reasoning models assess fact-checking claims under different evidence and prompt conditions. The project examines changes in generated thinking, verdicts, justifications, and self-reported confidence using AVeriTeC.

The current implementation runs **Qwen/Qwen3-4B-Thinking-2507** locally on a CUDA GPU. DeepSeek and GPT experiments are not implemented in this runner.

## Research questions

- How does supplied evidence change a model's verdict and justification?
- Does an explicitly empty evidence list behave differently from an omitted evidence field?
- When does a model generate answers based on apparent prior knowledge, and when does it abstain?
- How do prompt wording, evidence order, and missing evidence affect predictions?
- How do fresh post-verdict assessments differ when the earlier thinking trace is visible or hidden?

## What is recorded

The runner separates Qwen's generated thinking text, ending at `</think>`, from its final structured answer. The neutral confidence prompt requests a verdict, evidence IDs, justification, and confidence score.

**Thinking traces are generated text, not neural activations or a verified transcript of internal computation.** Activation recording and activation patching are not part of the current implementation. References to prior knowledge in a trace do not prove how the model internally reached its answer.

## Setup

Use Python 3.10+ and a BF16-capable CUDA GPU. The pilot runs used one NVIDIA L40S with approximately 48 GB VRAM; the working environment also had approximately 48 GB host RAM. GPU memory and host RAM are separate resources.

Install a CUDA-compatible PyTorch build appropriate for your cluster, then install the project dependencies:

```bash
python -m pip install -r requirements-experiment1.txt
```

The first model load downloads weights from Hugging Face unless they are already cached. Set `HF_HOME` to a persistent cache directory if needed. Use `--local-files-only` when running from an existing cache without network access.

The runner expects a labeled AVeriTeC JSON array, such as `train.json` or `dev.json`. Adjust `--data` if your cluster stores these under `Data/`. Model prompts contain the claim and annotated question-answer evidence, excluding dataset labels and dataset justifications.

## Run the experiment

Run this command inside your GPU allocation:

```bash
python experiment1_averitec.py \
  --data train.json \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --prompt-file prompts/neutral_confidence.json \
  --conditions gold no_evidence claim_only \
  --limit 10 --seed 42 --repeats 1 \
  --max-new-tokens 16384 \
  --skip-post-verdict \
  --output runs/neutral_10_claims
```

This selects ten shuffled claims and produces thirty stage-1 generations. Always use a **new, empty output directory**. Explicitly select the prompt file: the script's default is an evidence-only, verdict-only profile, not the neutral confidence profile.

Preview the prompts on a laptop without loading the model:

```bash
python experiment1_averitec.py \
  --data train.json --limit 1 \
  --prompt-file prompts/neutral_confidence.json \
  --conditions gold no_evidence claim_only \
  --skip-post-verdict --dry-run \
  --output runs/neutral_preview
```

Use `--repeats 3` for three sampling repetitions per claim and condition. Keep the dataset, seed, model revision, and generation settings fixed when comparing conditions. Pin `--revision` to a model commit SHA for reproducibility. Sampling uses temperature 0.6, top-p 0.95, and top-k 20.

## Balanced sampling

Add `--balanced` to sample evenly across the four dataset verdict labels:

```bash
python experiment1_averitec.py \
  --data train.json --balanced --limit 40 --seed 42 \
  --prompt-file prompts/neutral_confidence.json \
  --conditions gold no_evidence claim_only \
  --skip-post-verdict --output runs/balanced_40
```

This selects 10 claims per label, producing 120 stage-1 generations. For limits
not divisible by four, counts differ by at most one (10 claims gives 3/3/2/2);
the seed determines which labels receive the extra examples. Sampling is without
replacement and reproducible. `--offset` skips claims in the same balanced sequence,
so consecutive non-overlapping slices do not reuse claims. If a label has too few
examples for the requested offset and limit, the runner reports an error instead
of duplicating examples or silently changing the balance.

Without `--balanced`, the original shuffled sampling remains unchanged.
`metadata.json` records `selected_label_counts`. Accuracy on a balanced sample
describes that sample, not the dataset's original label distribution.

## Evidence conditions

| Condition | Model input | Purpose |
|---|---|---|
| `gold` | Claim and all annotated evidence | Provided-evidence baseline |
| `no_evidence` | Claim and `"evidence": []` | Explicitly empty evidence control |
| `claim_only` | Claim with no evidence field | Assessment without supplied evidence |
| `reversed` | Claim and evidence in reverse order | Evidence-order sensitivity |
| `drop_one` | Claim with one seeded QA-answer unit removed | Missing-evidence sensitivity |

All five conditions run if `--conditions` is omitted.

The neutral profile uses identical instructions across conditions, starting with “Evaluate the following fact-checking claim.” It neither prohibits nor explicitly encourages prior knowledge. For `no_evidence` versus `claim_only`, only the evidence field changes; the output schema still requests evidence IDs, with `[]` when none are supplied.

Editable prompt profiles live in [`prompts/`](prompts/README.md). They vary knowledge restrictions and final-answer requirements. The selected profile and its hash are saved with every run; verify these saved instructions when transferring changes to a cluster.

## Output files

| File | Contents |
|---|---|
| `results.jsonl` | One record per claim, condition, and repetition: exact prompt, `thinking`, `answer`, `prediction`, confidence, status, token counts, and optional post-verdict results |
| `readable/*.md` | Human-readable prompts, thinking traces, and final answers |
| `INDEX_rREPEAT_CONDITION.stage1.json` | Individual stage-1 checkpoint; not automatic resume support |
| `summary.json` | Verdict agreement, completion/truncation counts, and paired verdict changes |
| `metadata.json` | Arguments, sampled indices, hashes, software versions, and GPU details |
| `prompt_setting.json` | The selected prompt profile and provenance |

For example, open `readable/1258_r0_claim_only.md` to read that claim's trace, or inspect the `thinking` field in its JSON record. The `answer` field contains the final response after thinking.

Confidence is the model's **self-reported probability that its chosen label is correct**. A 100% confidence NEI answer expresses certainty about abstaining, not certainty that the claim is false. These scores are neither token probabilities nor measured repeat-run frequencies.

## Optional post-verdict comparison

Remove `--skip-post-verdict` to enable two fresh calls after each valid stage-1 answer:

- `without_trace`: provide the claim, condition-specific evidence input, and previous verdict, but hide the earlier thinking.
- `with_trace`: provide the same information plus the earlier thinking.

Both calls may retain or change the verdict and generate their own thinking and final justification. They do not resume the original hidden state. Agreement with an earlier trace does not establish faithful introspection. These calls can raise the workload to ninety generations for ten claims across three conditions.

## Initial ten-claim pilot

The inspected training-set pilot used one repetition per claim and the neutral confidence prompt. All thirty outputs completed without parsing errors or truncation.

| Measure | Gold evidence | Empty evidence array | Claim only |
|---|---:|---:|---:|
| Agreement with dataset labels | 8/10 | 0/10 | 2/10 |
| Not Enough Evidence verdicts | 0/10 | 10/10 | 4/10 |
| Mean output tokens | 2,118 | 854 | 3,270 |

Omitting the empty evidence field changed six of ten verdicts. Some traces mentioned background information but rejected its use when the evidence array was empty. Claim-only responses sometimes relied on apparent recollections, but also introduced unsupported assumptions or misread claims. Both gold-evidence errors were examples labeled Conflicting Evidence/Cherrypicking in the dataset.

These are exploratory observations from ten claims, not general performance estimates or a calibration study. None of the ten dataset labels was NEI, so the empty-array condition's zero agreement reflects uniform abstention.

## Evaluation and limitations

- This is an annotated-evidence experiment, **not the official AVeriTeC retrieval benchmark or score**. The runner does not search the web.
- Gold and reversed conditions use the original dataset label as reference. Under the neutral/unrestricted policy, no-evidence and claim-only outputs also use that label. Under evidence-only profiles, their protocol target is NEI. Drop-one has no assigned reference label.
- Correct labels do not guarantee sound explanations; incorrect labels can also expose ambiguity in claims or annotations. Review evidence and traces together.
- Claim dates, original images, and source context are not included in the current claim/evidence payload. Some claims require that context for meaningful interpretation.
- Repeated generations and conditions from the same claim are paired observations, not independent claims. Freeze prompts before held-out evaluation and inspect failures and truncation before scaling.

## Repository guide

- [`experiment1_averitec.py`](experiment1_averitec.py): main experiment runner.
- [`prompts/README.md`](prompts/README.md): prompt profiles and configuration.
- [`EXPERIMENT1.md`](EXPERIMENT1.md): detailed protocol and optional second-stage comparisons.
- [`extract_thinking.py`](extract_thinking.py): separate minimal extractor with its own fixed evidence-only prompt; it does not use the main runner's neutral profile.
- [`tests/test_experiment1.py`](tests/test_experiment1.py): protocol and parsing checks, including tiny-model generation.

Run the existing checks after installing dependencies:

```bash
python -m unittest discover -s tests -v
```