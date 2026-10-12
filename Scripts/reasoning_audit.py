#!/usr/bin/env python3
"""AVeriTeC prompt/evidence experiments with Qwen3 thinking traces.

See EXPERIMENT1.md. --dry-run requires only Python's standard library.
"""
import argparse
import hashlib
import json
import math
import platform
import random
import re
import time
from collections import Counter
from pathlib import Path

LABELS = ('Supported', 'Refuted', 'Not Enough Evidence',
          'Conflicting Evidence/Cherrypicking')
CONDITIONS = ('gold', 'no_evidence', 'claim_only', 'reversed', 'drop_one')
POST_MODES = ('without_trace', 'with_trace')
DEFAULT_PROMPT = Path(__file__).resolve().parent / 'prompts' / 'evidence_only_verdict.json'
DATA_MARKER = '\n\nINPUT_DATA_JSON:\n'


def load_prompt_setting(path=DEFAULT_PROMPT):
    path = Path(path)
    raw = path.read_bytes()
    setting = json.loads(raw)
    if not isinstance(setting, dict):
        raise ValueError('Prompt file must contain a JSON object')
    if setting.get('knowledge_policy') not in ('evidence_only', 'unrestricted', 'prior_allowed'):
        raise ValueError('knowledge_policy must be evidence_only, unrestricted, or prior_allowed')
    if setting.get('answer_format') not in ('verdict', 'justification', 'confidence'):
        raise ValueError('answer_format must be verdict, justification, or confidence')
    keys = ['name', 'stage1_instructions', 'post_instructions']
    keys += [k for k in ('claim_only_instructions', 'claim_only_post_instructions') if k in setting]
    for key in keys:
        if not isinstance(setting.get(key), str) or not setting[key].strip():
            raise ValueError(f'Prompt file requires a nonempty {key}')
        if DATA_MARKER in setting[key]:
            raise ValueError('Prompt instructions must not contain the reserved input-data marker')
    return {**setting, 'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def evaluation_target(record, condition, setting):
    if condition in ('gold', 'reversed'):
        return record['label'], 'dataset_label'
    if condition in ('no_evidence', 'claim_only'):
        if setting['knowledge_policy'] == 'evidence_only':
            return 'Not Enough Evidence', 'closed_evidence_protocol'
        return record['label'], 'dataset_label_without_supplied_evidence'
    return None, 'unassigned_modified_evidence'


def assemble_prompt(instructions, payload):
    return instructions + DATA_MARKER + json.dumps(payload, ensure_ascii=False)


def prompt_payload(prompt):
    return json.loads(prompt.split(DATA_MARKER, 1)[1])


def evidence_units(record):
    """Allowlist prompt fields; never include the verdict or justification."""
    units = []
    for qi, question in enumerate(record.get('questions', [])):
        for ai, answer in enumerate(question.get('answers', [])):
            if not isinstance(answer.get('answer'), str) or not answer['answer'].strip():
                continue
            unit = {'id': f'Q{qi + 1}A{ai + 1}', 'question': question['question'],
                    'answer': answer['answer']}
            if answer.get('boolean_explanation'):
                unit['boolean_explanation'] = answer['boolean_explanation']
            units.append(unit)
    return units


def make_prompt(record, condition, seed, setting=None):
    setting = setting or load_prompt_setting()
    units = evidence_units(record)
    original_ids = [u['id'] for u in units]
    if condition in ('no_evidence', 'claim_only'):
        units = []
    elif condition == 'reversed':
        units = units[::-1]
    elif condition == 'drop_one' and units:
        units = units.copy()
        units.pop(random.Random(seed).randrange(len(units)))
    elif condition != 'gold' and condition not in CONDITIONS:
        raise ValueError(condition)
    # URLs are deliberately omitted: URL slugs can reveal the verdict.
    payload = {'claim': record['claim']}
    instructions = setting['stage1_instructions']
    if condition == 'claim_only':
        instructions = setting.get('claim_only_instructions', instructions)
    else:
        payload['evidence'] = units
    prompt = assemble_prompt(instructions, payload)
    return prompt, original_ids, [u['id'] for u in units]


def make_post_prompt(prompt, result, mode, setting=None):
    """Fresh post-verdict call; only trace visibility differs between arms."""
    if mode not in POST_MODES:
        raise ValueError(mode)
    payload = prompt_payload(prompt)
    # Do not pass original evidence_ids: the verdict alone is shared across arms.
    payload['previous_verdict'] = result['prediction']
    payload['earlier_thinking'] = result['thinking'] if mode == 'with_trace' else None
    setting = setting or load_prompt_setting()
    instructions = setting['post_instructions']
    if 'evidence' not in payload:
        instructions = setting.get('claim_only_post_instructions', instructions)
    return assemble_prompt(instructions, payload)


def parse_output(text, stage='verdict', answer_format='verdict'):
    # This thinking-only checkpoint puts the opening <think> in the prompt.
    if '</think>' not in text:
        return {'thinking': text, 'answer': '', 'prediction': None,
                'parse_error': 'missing_thinking_end'}
    thinking, answer = text.rsplit('</think>', 1)
    answer = answer.strip()
    candidate = re.sub(r'^```(?:json)?\s*|\s*```$', '', answer).strip()
    try:
        obj = json.loads(candidate)
        valid = (isinstance(obj, dict) and obj.get('label') in LABELS
                 and isinstance(obj.get('evidence_ids'), list)
                 and all(isinstance(x, str) for x in obj['evidence_ids'])
                 and (not (stage == 'post_verdict' or answer_format in ('justification', 'confidence'))
                      or (isinstance(obj.get('explanation'), str) and bool(obj['explanation'].strip()))))
        if valid and answer_format == 'confidence':
            score = obj.get('confidence')
            valid = (type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1)
    except (ValueError, TypeError):
        obj, valid = None, False
    return {'thinking': thinking.removeprefix('<think>').strip(), 'answer': answer,
            'prediction': obj['label'] if valid else None,
            'confidence': obj.get('confidence') if valid and answer_format == 'confidence' else None,
            'answer_object': obj, 'parse_error': None if valid else 'invalid_answer_json'}


def generate(model, tokenizer, prompt, args, seed, stage='verdict'):
    import torch
    from transformers import set_seed
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
    n_input = inputs.input_ids.shape[1]
    input_limit = args.max_post_input_tokens if stage == 'post_verdict' else args.max_input_tokens
    output_limit = args.max_post_tokens if stage == 'post_verdict' else args.max_new_tokens
    if n_input > input_limit:
        return {'status': 'skipped_input_too_long', 'input_tokens': n_input}
    inputs = inputs.to(model.device)
    set_seed(seed)
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=output_limit, use_cache=True,
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
            num_beams=1, num_return_sequences=1,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id))
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    ids = output[0, n_input:].tolist()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    reached_eos = bool(ids and ids[-1] in eos)
    # Remove only terminal EOS; preserve the thinking boundary for parsing.
    text = tokenizer.decode(ids[:-1] if reached_eos else ids, skip_special_tokens=False)
    setting = getattr(args, 'prompt_setting', None) or load_prompt_setting()
    result = parse_output(text, stage, setting['answer_format'])
    result.update(status='ok', stage=stage, input_tokens=n_input, output_tokens=len(ids),
                  truncated=not reached_eos and len(ids) >= output_limit,
                  raw_output=text, seconds=elapsed,
                  peak_gpu_allocated_gib=(torch.cuda.max_memory_allocated() / 2**30
                                          if model.device.type == 'cuda' else None))
    return result


def complete_result(row):
    return (row.get('status') == 'ok' and not row.get('truncated', True)
            and row.get('prediction') is not None)


def compare_explanation(row, post):
    """Descriptive ID overlap, not a semantic/causal faithfulness score."""
    if not complete_result(post):
        return None
    trace_ids = set(re.findall(r'\bQ\d+A\d+\b', row.get('thinking', '')))
    citations = set(post['answer_object']['evidence_ids'])
    retained = set(row['retained_evidence_ids'])
    return {'verdict_changed': post['prediction'] != row['prediction'],
            'trace_explicit_evidence_ids': sorted(trace_ids),
            'explanation_cited_ids': sorted(citations),
            'trace_ids_not_cited_afterward': sorted(trace_ids - citations),
            'explanation_ids_not_explicit_in_trace': sorted(citations - trace_ids),
            'unknown_cited_ids': sorted(citations - retained),
            'interpretation': 'Literal ID overlap only. Evidence can be discussed without IDs; '
                              'missing IDs do not establish omitted reasoning or unused evidence.'}


def run_post_verdict(model, tokenizer, row, args):
    if not complete_result(row):
        return {mode: {'status': 'skipped_incomplete_verdict'} for mode in args.post_modes}
    posts = {}
    for mode in dict.fromkeys(args.post_modes):
        prompt = make_post_prompt(row['prompt'], row, mode, getattr(args, 'prompt_setting', None))
        # Common random seed across post arms; separate from the original run.
        seed = row['seed'] + 10_000_000
        post = generate(model, tokenizer, prompt, args, seed, stage='post_verdict')
        post.update(prompt=prompt, seed=seed, trace_visibility=mode,
                    conditioning='fresh_call_with_original_evidence_and_previous_verdict')
        post['comparison'] = compare_explanation(row, post)
        posts[mode] = post
    return posts


def write_readable(row, path):
    """A per-run report so long JSON lines never hide the thinking field."""
    parts = [f'# Claim {row["dataset_index"]}: {row["condition"]}, repeat {row["repeat"]}',
             'This report contains generated thinking traces and answers.',
             f'Prompt setting: {row.get("prompt_setting", "legacy")}; '
             f'knowledge policy: {row.get("knowledge_policy", "legacy")}; '
             f'answer format: {row.get("answer_format", "legacy")}',
             f'Run status: {row["status"]}; truncated: {row.get("truncated", "not run")}',
             '## Stage 1 prompt', row['prompt'], '## Stage 1: thinking before verdict',
             row.get('thinking') or '(No generated thinking. Check run status.)',
             '## Stage 1: committed verdict', row.get('answer') or '(No completed answer.)']
    for mode, post in row.get('post_verdict', {}).items():
        parts += [f'## Stage 2: {mode}',
                  f'Status: {post["status"]}; truncated: {post.get("truncated", "not run")}',
                  '### Post-verdict prompt', post.get('prompt', '(Stage skipped.)'),
                  '### New thinking generated after the verdict',
                  post.get('thinking') or '(No generated thinking.)',
                  '### Post-verdict explanation', post.get('answer') or '(No completed explanation.)',
                  '### Descriptive comparison', json.dumps(post.get('comparison'), indent=2)]
    parts += ['## Human review (not automatically inferred)',
              '- Which evidence is discussed before versus after the verdict?\n'
              '- Are uncertainty, alternatives, or corrections preserved?\n'
              '- Does the explanation introduce claims unsupported by the supplied evidence?\n'
              '- Record exact quotations and evidence IDs; do not infer absence from an omitted ID.']
    path.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')


def summarize(rows):
    report = {}
    for condition in sorted({r['condition'] for r in rows}):
        group = [r for r in rows if r['condition'] == condition]
        eligible = [r for r in group if r['status'] == 'ok' and not r['truncated']
                    and r.get('prediction') is not None]
        scores = [r for r in group if r.get('evaluation_label') is not None
                  and r['status'] != 'dry_run']
        correct = sum(r.get('prediction') == r['evaluation_label']
                      and r['status'] == 'ok' and not r.get('truncated', True)
                      for r in scores)
        report[condition] = {'attempts': len(group), 'complete_valid_verdicts': len(eligible),
                             'status_counts': dict(Counter(r['status'] for r in group)),
                             'truncated': sum(r.get('truncated', False) for r in group),
                             'accuracy_all_attempts': correct / len(scores) if scores else None}
    indexed = {(r['dataset_index'], r['repeat'], r['condition']): r for r in rows}
    pairs = {}
    for condition in report:
        if condition == 'gold':
            continue
        valid_pairs = []
        for row in rows:
            if row['condition'] != condition:
                continue
            base = indexed.get((row['dataset_index'], row['repeat'], 'gold'))
            if base and all(x['status'] == 'ok' and not x.get('truncated', True)
                            and x.get('prediction') is not None for x in (base, row)):
                valid_pairs.append(base['prediction'] != row['prediction'])
        pairs[condition] = {'valid_pairs': len(valid_pairs),
                            'verdict_flip_rate_vs_gold': (sum(valid_pairs) / len(valid_pairs)
                                                          if valid_pairs else None)}
    post_summary = {}
    for mode in POST_MODES:
        attempts = [r['post_verdict'][mode] for r in rows if mode in r.get('post_verdict', {})]
        valid = [x for x in attempts if complete_result(x)]
        changes = [x['comparison']['verdict_changed'] for x in valid]
        post_summary[mode] = {
            'planned': len(attempts), 'status_counts': dict(Counter(x['status'] for x in attempts)),
            'complete_explanations': len(valid),
            'truncated': sum(x.get('truncated', False) for x in attempts),
            'verdict_change_rate': sum(changes) / len(changes) if changes else None}
    return {'by_condition': report, 'paired_comparisons': pairs,
            'post_verdict': post_summary,
            'note': 'Gold-evidence verdict accuracy, not the official AVeriTeC score. '
                    'Check evaluation_basis: no_evidence/claim_only use NEI only under evidence_only; '
                    'other knowledge policies use the original dataset label as reference. '
                    'Drop-one conditions have no assigned gold label.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('train.json'))
    p.add_argument('--output', type=Path, required=True, help='New, empty run directory')
    p.add_argument('--prompt-file', type=Path, default=DEFAULT_PROMPT,
                   help='Editable JSON prompt setting (see prompts/README.md)')
    p.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507')
    p.add_argument('--revision', default='main', help='Use a commit SHA for repeatable runs')
    p.add_argument('--limit', type=int, default=5)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--max-input-tokens', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=16384)
    p.add_argument('--max-post-input-tokens', type=int, default=24576)
    p.add_argument('--max-post-tokens', type=int, default=8192)
    p.add_argument('--post-modes', nargs='+', choices=POST_MODES, default=list(POST_MODES))
    p.add_argument('--skip-post-verdict', action='store_true', help='Run only stage 1')
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Prepare prompts without loading a model')
    args = p.parse_args()
    try:
        setting = load_prompt_setting(args.prompt_file)
    except (OSError, ValueError) as error:
        p.error(f'Invalid prompt file: {error}')
    if min(args.limit, args.repeats, args.max_input_tokens,
           args.max_new_tokens, args.max_post_input_tokens, args.max_post_tokens) < 1 or args.offset < 0:
        p.error('Counts and token limits must be positive; offset must be nonnegative.')
    data_bytes = args.data.read_bytes()
    data = json.loads(data_bytes)
    if not isinstance(data, list) or not data:
        p.error('Expected a nonempty AVeriTeC JSON array.')
    for row in data:
        if not isinstance(row.get('claim'), str) or row.get('label') not in LABELS:
            p.error('Expected labeled AVeriTeC train/dev records with claim and label.')
    # Random sample avoids a first-N pilot tied to the dataset's ordering.
    indices = list(range(len(data)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[args.offset:args.offset + args.limit]
    if not indices:
        p.error('Selection is empty.')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory must be empty; use a new run name to avoid mixing runs.')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 4,
                'prompt_setting': setting,
                'protocol': 'verdict_then_fresh_post_verdict_explanation_with_trace_visibility_control',
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selected_indices': indices, 'python': platform.python_version(),
                'dataset_label_counts': dict(Counter(r['label'] for r in data))}
    # Keep this out of the CLI-arguments snapshot; the full setting is saved above.
    args.prompt_setting = setting
    (args.output / 'prompt_setting.json').write_text(json.dumps(setting, indent=2) + '\n')
    model = tokenizer = None
    if not args.dry_run:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            p.error('A CUDA GPU is required for this runner; use --dry-run on your laptop.')
        if not torch.cuda.is_bf16_supported():
            p.error('Use an L40S/A100/H100 or other BF16-capable GPU for this BF16 protocol.')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  local_files_only=args.local_files_only, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
        model.eval()
        if args.max_input_tokens + args.max_new_tokens > model.config.max_position_embeddings:
            p.error('Requested total token budget exceeds the model context window.')
        if not args.skip_post_verdict and args.max_post_input_tokens + args.max_post_tokens > model.config.max_position_embeddings:
            p.error('Requested post-verdict token budget exceeds the model context window.')
        metadata.update(torch=torch.__version__, transformers=transformers.__version__,
                        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                        resolved_model_revision=getattr(model.config, '_commit_hash', None))
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'readable').mkdir()
    results = []
    with (args.output / 'results.jsonl').open('w') as out:
        def run_one(index, repeat, condition):
            record = data[index]
            seed = args.seed + index * 1009 + repeat
            prompt, original, retained = make_prompt(record, condition, args.seed + index, setting)
            target, basis = evaluation_target(record, condition, setting)
            row = dict(dataset_index=index, repeat=repeat, condition=condition, seed=seed,
                       dataset_label=record['label'], evaluation_label=target, prompt=prompt,
                       evaluation_basis=basis, prompt_setting=setting['name'],
                       prompt_sha256=setting['sha256'], knowledge_policy=setting['knowledge_policy'],
                       answer_format=setting['answer_format'],
                       original_evidence_ids=original, retained_evidence_ids=retained,
                       evidence_changed=retained != original)
            if args.dry_run:
                result = {'status': 'dry_run'}
            else:
                result = generate(model, tokenizer, prompt, args, seed)
            row.update(result)
            if row.get('prediction'):
                row['unknown_cited_ids'] = sorted(set(row['answer_object']['evidence_ids']) - set(retained))
            # Checkpoint stage 1 immediately, even if a later explanation fails.
            checkpoint = args.output / f'{index}_r{repeat}_{condition}.stage1.json'
            checkpoint.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
            readable_path = args.output / 'readable' / f'{index}_r{repeat}_{condition}.md'
            write_readable(row, readable_path)
            if not args.skip_post_verdict:
                if args.dry_run:
                    row['post_verdict'] = {mode: {'status': 'dry_run_requires_generated_verdict'}
                                           for mode in args.post_modes}
                else:
                    print(f'claim={index} condition={condition}: starting post-verdict stage', flush=True)
                    row['post_verdict'] = run_post_verdict(model, tokenizer, row, args)
            row['readable_file'] = str(readable_path)
            write_readable(row, readable_path)
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            results.append(row)
            (args.output / 'summary.json').write_text(json.dumps(summarize(results), indent=2) + '\n')
            print(f'claim={index} repeat={repeat} condition={condition} '
                  f'status={row["status"]} verdict={row.get("prediction")} '
                  f'truncated={row.get("truncated")}', flush=True)

        for index in indices:
            for repeat in range(args.repeats):
                for condition in dict.fromkeys(args.conditions):
                    run_one(index, repeat, condition)
    print(f'Results saved to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
