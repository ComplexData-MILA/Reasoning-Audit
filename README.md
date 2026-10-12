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
python -m pip install -r requirements.txt
```

The first model load downloads weights from Hugging Face unless they are already cached. Set `HF_HOME` to a persistent cache directory if needed. Use `--local-files-only` when running from an existing cache without network access.

The runner expects a labeled AVeriTeC JSON array, such as `train.json` or `dev.json`. Adjust `--data` if your cluster stores these under `Data/`. Model prompts contain the claim and annotated question-answer evidence, excluding dataset labels and dataset justifications.

## Run the experiment

Run this command inside your GPU allocation:

```bash
python experiment1_averitec.py \
  --data train.json \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --prompt-file prompts/default_prompt.json \
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
  --prompt-file prompts/default_prompt.json \
  --conditions gold no_evidence claim_only \
  --skip-post-verdict --dry-run \
  --output runs/neutral_preview
```

Use `--repeats 3` for three sampling repetitions per claim and condition. Keep the dataset, seed, model revision, and generation settings fixed when comparing conditions. Pin `--revision` to a model commit SHA for reproducibility. Sampling uses temperature 0.6, top-p 0.95, and top-k 20.

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
