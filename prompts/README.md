# Prompt settings

Choose one editable JSON file per run using `--prompt-file`. Each file contains
the complete stage-1 and optional post-verdict instructions. The script appends
the claim/evidence automatically; never paste dataset verdicts into instructions.

| File | Knowledge policy | Final stage-1 output |
|---|---|---|
| evidence_only_verdict.json | Supplied evidence only | Verdict + evidence IDs |
| evidence_only_justification.json | Supplied evidence only | Verdict + IDs + justification |
| evidence_only_confidence.json | Supplied evidence only | Verdict + IDs + justification + confidence |
| unrestricted_verdict.json | No restriction on outside knowledge | Verdict + evidence IDs |
| unrestricted_justification.json | No restriction on outside knowledge | Verdict + IDs + justification |
| unrestricted_confidence.json | No restriction on outside knowledge | Verdict + IDs + justification + confidence |
| default_prompt.json | No mention of prior/outside knowledge | Verdict + IDs + justification + confidence |

`unrestricted` removes both the outside-knowledge prohibition and the instruction
to use *only* supplied evidence. It does not explicitly encourage prior knowledge.
The label definitions are identical across the profiles. If you want to
explicitly permit prior knowledge, copy a file, edit BOTH stages' instructions,
set `knowledge_policy` to `prior_allowed`, and give it a unique `name`.

`default_prompt.json` includes no instruction about prior knowledge, outside
knowledge, or filling gaps. It asks for an assessment, justification, and confidence
without specifying where the model must obtain its information. The unrestricted
policy in the JSON metadata controls evaluation and is not sent to the model.

## Example: justification and confidence, no knowledge restriction

```bash
python experiment1_averitec.py --data train.json --limit 5 \
  --conditions gold no_evidence \
  --prompt-file prompts/unrestricted_confidence.json \
  --skip-post-verdict --output runs/unrestricted_confidence
```

This requests the justification and confidence **in the first call**, following
the thinking trace. A second call is not needed. Without `--skip-post-verdict`,
the selected file's `post_instructions` also apply to the later explanation calls.
Confidence is required in both stages for a confidence profile.

Example final answer:

```json
{
  "label": "Supported",
  "evidence_ids": ["Q1A1"],
  "explanation": "The supplied evidence supports the claim because ...",
  "confidence": 0.82
}
```

Confidence means the model's self-reported probability that its **chosen label
is correct**, not the probability that the claim is true. A model can confidently
choose NEI. This number is not a token probability and is not guaranteed calibrated.
The parser requires a finite numeric value in [0,1]; strings, booleans, missing
values and out-of-range scores invalidate the structured answer. It also requires
a nonempty `explanation` for justification/confidence profiles.

## Compare settings fairly

Use the same data, seed, limit, offset, repeats, model revision and token budgets
in separate output directories. The script will select the same claim indices
and sampling seeds. Change only one factor at a time: knowledge policy OR output
format. Asking for confidence/justification can change the thinking itself; it
is not merely an extra measurement of an unchanged process.

Preview without a GPU by adding `--dry-run` to a command with a fresh directory.
The actual stage-1 prompts appear in `readable/`. Post-verdict prompts depend on
the generated verdict and are constructed only in real runs.

## Evaluation and provenance

- Evidence-only + no evidence: target is protocol-defined NEI.
- Unrestricted/prior-allowed + no evidence: reference is the original dataset
  label, not automatically NEI. Correctness does not establish knowledge use;
  it may reflect memorization or guessing. NEI may still occur.
- Gold/reversed evidence: original dataset label.
- Drop-one conditions: no assigned target without further annotation.

The result's `evaluation_basis` identifies the rule used. Do not pool accuracy
across knowledge policies without noting this target difference.

The exact file contents/settings and SHA-256 are saved in `prompt_setting.json`
and `metadata.json`. Each result records the profile name, hash, knowledge policy,
answer format, exact prompt, and extracted confidence (when requested). Readable
reports show the complete final answer, including justification/confidence.

Edit instruction strings freely, but keep `answer_format` synchronized with the
requested schema: `verdict`, `justification`, or `confidence`. Metadata does not
automatically rewrite instructions. The reserved `INPUT_DATA_JSON:` separator
marks where the generated payload starts; do not place it in your instructions.
Keep dataset evidence out of the static prompt file.

The default remains evidence_only_verdict. Version 3 prompts use shared label
definitions and a new input delimiter, so treat these as new experiments rather
than exact replications of older runs. Existing run files are not modified.
