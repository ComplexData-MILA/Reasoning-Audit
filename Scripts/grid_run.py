#!/usr/bin/env python3
"""Run a balanced claim x evidence x prompt x sampling grid (stage 1 only)."""
import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

from reasoning_audit import (
    DATA_MARKER, LABELS, assemble_prompt, complete_result, evidence_units,
    generate, load_prompt_setting, select_indices, write_readable,
)

ROOT = Path(__file__).resolve().parent.resolve().parent
CONDITIONS = ('claim_only', 'single_evidence', 'full_evidence', 'full_plus_contradiction')
DEFAULT_PROMPTS = [ROOT / 'prompts' / 'grid' / f'{name}.json'
                   for name in ('neutral', 'evidence_focused', 'prior_allowed', 'step_by_step')]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_evidence_manifest(path, data, data_bytes, indices):
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('dataset_sha256') != digest(data_bytes):
        raise ValueError('Evidence manifest dataset hash differs from --data.')
    entries = manifest['entries']
    indexed = {e['dataset_index']: e for e in entries}
    if len(indexed) != len(entries) or set(indexed) != set(indices):
        raise ValueError('Manifest must contain exactly one entry per selected claim. '
                         'Use its original selection settings or prepare a matching manifest.')
    for index, entry in indexed.items():
        units = {u['id']: u for u in evidence_units(data[index])}
        source = units.get(entry['single_evidence_id'])
        if entry['claim'] != data[index]['claim'] or source != entry['source_unit']:
            raise ValueError(f'Manifest claim/source mismatch at index {index}.')
        answer = entry.get('synthetic_answer')
        if not isinstance(answer, str) or not answer.strip() or answer.strip() == source['answer'].strip():
            raise ValueError(f'Missing or unchanged synthetic answer at index {index}.')
        if DATA_MARKER in answer:
            raise ValueError('Synthetic answer contains reserved input delimiter.')
    return manifest, indexed, digest(raw)


def grid_payload(record, condition, entry):
    units = evidence_units(record)
    payload = {'claim': record['claim']}
    if condition == 'claim_only':
        return payload
    if condition == 'single_evidence':
        units = [u for u in units if u['id'] == entry['single_evidence_id']]
        if len(units) != 1:
            raise ValueError('Single evidence ID not found.')
    elif condition == 'full_plus_contradiction':
        # Ordinary QA-shaped ID; synthetic provenance lives outside the prompt.
        source = entry['source_unit']
        synthetic = {'id': f'Q{len(record.get("questions", [])) + 1}A1',
                     'question': source['question'], 'answer': entry['synthetic_answer']}
        units = units + [synthetic]
    elif condition != 'full_evidence':
        raise ValueError(condition)
    payload['evidence'] = units
    return payload


def build_plan(data, indices, entries, settings, repeats, seed):
    rows = []
    for index in indices:
        for setting in settings:
            for repeat in range(repeats):
                for condition in CONDITIONS:
                    record = data[index]
                    payload = grid_payload(record, condition, entries[index])
                    modified = condition in ('single_evidence', 'full_plus_contradiction')
                    synthetic_id = (payload['evidence'][-1]['id']
                                    if condition == 'full_plus_contradiction' else None)
                    rows.append(dict(
                        run_id=f'{index}__{setting["name"]}__r{repeat}__{condition}',
                        dataset_index=index, repeat=repeat, condition=condition,
                        seed=seed + index * 1009 + repeat,
                        dataset_label=record['label'],
                        evaluation_label=None if modified else record['label'],
                        evaluation_basis=('unassigned_modified_evidence' if modified else
                                          'dataset_label' if condition == 'full_evidence' else
                                          'dataset_label_without_supplied_evidence'),
                        prompt=assemble_prompt(setting['stage1_instructions'], payload),
                        prompt_setting=setting['name'], prompt_sha256=setting['sha256'],
                        knowledge_policy=setting['knowledge_policy'], answer_format='confidence',
                        original_evidence_ids=[u['id'] for u in evidence_units(record)],
                        retained_evidence_ids=[u['id'] for u in payload.get('evidence', [])],
                        evidence_changed=condition != 'full_evidence',
                        single_evidence_id=entries[index]['single_evidence_id'],
                        synthetic_evidence_id=synthetic_id,
                        synthetic_source_id=entries[index]['single_evidence_id'] if synthetic_id else None,
                        synthetic_review_status=entries[index].get('review_status', 'unspecified'),
                    ))
    return rows


def summarize_grid(rows, planned):
    cells = defaultdict(list)
    for row in rows:
        cells[(row['prompt_setting'], row['condition'])].append(row)
    report = {}
    index = {(r['dataset_index'], r['repeat'], r['prompt_setting'], r['condition']): r
             for r in rows}
    for (prompt, condition), group in sorted(cells.items()):
        valid = [r for r in group if complete_result(r)]
        scored = [r for r in group if r.get('evaluation_label') is not None and r['status'] != 'dry_run']
        paired = [(r, index.get((r['dataset_index'], r['repeat'], prompt, 'full_evidence')))
                  for r in valid]
        paired = [(r, base) for r, base in paired if base and complete_result(base)]
        by_claim = defaultdict(list)
        for r in valid:
            by_claim[r['dataset_index']].append(r)
        triples = [g for g in by_claim.values() if len(g) == planned['repeats']]
        synth = [r for r in valid if r['synthetic_evidence_id']]
        report.setdefault(prompt, {})[condition] = {
            'attempts': len(group), 'complete_valid_verdicts': len(valid),
            'status_counts': dict(Counter(r['status'] for r in group)),
            'truncated': sum(bool(r.get('truncated')) for r in group),
            'parse_failures': sum(bool(r.get('parse_error')) for r in group),
            'unknown_citation_runs': sum(bool(r.get('unknown_cited_ids')) for r in group),
            'prediction_counts': dict(Counter(r['prediction'] for r in valid)),
            'accuracy_all_attempts': (sum(complete_result(r) and r['prediction'] == r['evaluation_label']
                                          for r in scored) / len(scored) if scored else None),
            'dataset_label_agreement_valid': (sum(r['prediction'] == r['dataset_label'] for r in valid)
                                              / len(valid) if valid else None),
            'mean_confidence_valid': sum(r['confidence'] for r in valid) / len(valid) if valid else None,
            'valid_pairs_vs_full': len(paired),
            'verdict_flip_rate_vs_full': sum(r['prediction'] != b['prediction'] for r, b in paired)
                                        / len(paired) if paired else None,
            'claims_with_all_samples_valid': len(triples),
            'claims_with_verdict_variation': sum(len({r['prediction'] for r in g}) > 1 for g in triples),
            'synthetic_citation_rate_valid': (sum(r['synthetic_evidence_id'] in r['answer_object']['evidence_ids']
                                                  for r in synth) / len(synth) if synth else None),
        }
    return {'planned': planned, 'recorded': len(rows), 'by_prompt_and_condition': report,
            'note': 'Modified evidence has no automatic correctness target. Dataset-label agreement '
                    'is descriptive there. Synthetic citation is self-report, not causal attribution. '
                    'Samples are paired within claims; they are not independent claims.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('train.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence-manifest', type=Path, default=ROOT / 'experiments/grid40_evidence.json')
    p.add_argument('--prompt-files', nargs=4, type=Path, default=DEFAULT_PROMPTS)
    p.add_argument('--limit', type=int, default=40)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507')
    p.add_argument('--revision', default='main')
    p.add_argument('--max-input-tokens', type=int, default=8192)
    p.add_argument('--max-new-tokens', type=int, default=16384)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true', help='Reuse checkpoints only with identical inputs/settings')
    args = p.parse_args()
    if min(args.limit, args.repeats, args.max_input_tokens, args.max_new_tokens) < 1 or args.offset < 0:
        p.error('Counts/budgets must be positive and offset nonnegative.')
    try:
        data_bytes = args.data.read_bytes()
        data = json.loads(data_bytes)
        if not isinstance(data, list) or not data or any(
                not isinstance(r, dict) or r.get('label') not in LABELS
                or not isinstance(r.get('claim'), str) for r in data):
            raise ValueError('Expected labeled AVeriTeC JSON records.')
        indices = select_indices(data, args.seed, args.offset, args.limit, balanced=True)
        manifest, entries, manifest_hash = load_evidence_manifest(args.evidence_manifest, data, data_bytes, indices)
        settings = [load_prompt_setting(f) for f in args.prompt_files]
        names = [s['name'] for s in settings]
        if len(set(names)) != 4 or any(not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name) for name in names):
            raise ValueError('Four distinct prompt names using lowercase letters, digits, _ or - required.')
        if any(s['answer_format'] != 'confidence' or 'claim_only_instructions' in s for s in settings):
            raise ValueError('Grid prompts must use confidence format and shared instructions across conditions.')
        plan = build_plan(data, indices, entries, settings, args.repeats, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.error(str(error))
    planned = {'claims': len(indices), 'conditions': len(CONDITIONS), 'prompts': len(settings),
               'repeats': args.repeats, 'generations': len(plan)}
    provenance = {
        'dataset_sha256': digest(data_bytes), 'evidence_manifest_sha256': manifest_hash,
        'grid_script_sha256': digest(Path(__file__).read_bytes()),
        'runner_script_sha256': digest((ROOT / 'Scripts/reasoning_audit.py').read_bytes()),
        'arguments': {k: ([str(x) for x in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items() if k not in ('resume', 'output')},
        'prompt_hashes': {s['name']: s['sha256'] for s in settings},
    }
    fingerprint = digest(json.dumps(provenance, sort_keys=True).encode())
    metadata_path = args.output / 'metadata.json'
    if args.resume:
        if not metadata_path.exists():
            p.error('--resume requires an existing grid metadata.json.')
        metadata = json.loads(metadata_path.read_text())
        if metadata.get('fingerprint') != fingerprint:
            p.error('Resume rejected: inputs, prompts, code or settings changed. Use a new output directory.')
    else:
        if args.output.exists() and any(args.output.iterdir()):
            p.error('Output must be empty unless --resume is used.')
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = dict(schema_version=1, fingerprint=fingerprint, provenance=provenance,
                        planned=planned, selected_indices=indices,
                        selected_label_counts=dict(Counter(data[i]['label'] for i in indices)),
                        python=platform.python_version(),
                        synthetic_position='appended_last',
                        single_selection='fixed manually selected source QA from evidence manifest',
                        conditions=list(CONDITIONS), prompt_settings=settings)
        atomic_json(metadata_path, metadata)
        atomic_json(args.output / 'evidence_manifest.json', manifest)
        atomic_json(args.output / 'prompt_settings.json', settings)
    checkpoints = args.output / 'checkpoints'
    readable = args.output / 'readable'
    checkpoints.mkdir(exist_ok=True)
    readable.mkdir(exist_ok=True)
    results = []
    finished = set()
    for row in plan:
        path = checkpoints / f'{row["run_id"]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if any(saved.get(k) != v for k, v in row.items()) or saved.get('fingerprint') != fingerprint:
                p.error(f'Checkpoint mismatch: {path}')
            if saved.get('status') not in ('ok', 'dry_run', 'skipped_input_too_long'):
                p.error(f'Unrecognized checkpoint status: {path}')
            results.append(saved)
            finished.add(row['run_id'])
            write_readable(saved, readable / f'{row["run_id"]}.md')
    model = tokenizer = None
    if not args.dry_run and len(finished) < len(plan):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            p.error('A BF16-capable CUDA GPU is required; use --dry-run for a CPU preview.')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  local_files_only=args.local_files_only, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only,
            torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
        model.eval()
        if args.max_input_tokens + args.max_new_tokens > model.config.max_position_embeddings:
            p.error('Input plus output budget exceeds the model context window.')
        environment = dict(torch=torch.__version__, transformers=transformers.__version__,
                           cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                           resolved_model_revision=getattr(model.config, '_commit_hash', None))
        if args.resume and metadata.get('environment') and metadata['environment'] != environment:
            p.error('Resume environment/model revision differs from the original run.')
        metadata['environment'] = environment
        atomic_json(metadata_path, metadata)
    print(f'Grid: {len(plan)} generations; {len(finished)} checkpoints present. '
          'Synthetic evidence provenance is saved separately from model input.', flush=True)
    # Rebuild the aggregate from atomic checkpoints, recovering interrupted JSONL writes.
    with (args.output / 'results.jsonl').open('w') as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
        out.flush()
        atomic_json(args.output / 'summary.json', summarize_grid(results, planned))
        for row in plan:
            if row['run_id'] in finished:
                continue
            args.prompt_setting = next(s for s in settings if s['name'] == row['prompt_setting'])
            result = {'status': 'dry_run'} if args.dry_run else generate(
                model, tokenizer, row['prompt'], args, row['seed'])
            row = {**row, **result, 'fingerprint': fingerprint}
            if row.get('prediction'):
                row['unknown_cited_ids'] = sorted(set(row['answer_object']['evidence_ids'])
                                                  - set(row['retained_evidence_ids']))
            row['readable_file'] = str(readable / f'{row["run_id"]}.md')
            atomic_json(checkpoints / f'{row["run_id"]}.json', row)
            write_readable(row, Path(row['readable_file']))
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            results.append(row)
            atomic_json(args.output / 'summary.json', summarize_grid(results, planned))
            print(f'{len(results)}/{len(plan)} {row["run_id"]}: '
                  f'{row["status"]} {row.get("prediction")}', flush=True)
    print(f'Results saved to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
