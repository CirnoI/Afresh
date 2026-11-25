"""Read published one-shot results only after their completion marker commits."""
from __future__ import annotations

import json
from pathlib import Path

from .paths import resolve_manifest_path
from .tsv import read_tsv


def completed_run(root: Path, run_id: str) -> dict:
    receipt_path = root / "one_shot_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "iils-one-shot-v1"
            or receipt.get("run_id") != run_id
            or receipt.get("status") != "completed"):
        raise ValueError(f"{root}: expected completed one-shot result for {run_id}")
    return receipt


def result_rows(root: Path, manifest: Path, policy: str, run_ids=None) -> list[dict]:
    """Attach published final paths to explicit experiment metadata in memory."""
    selected = set(run_ids or [])
    rows = read_tsv(manifest, required={'run_id'})
    result = []
    for row in rows:
        run_id = row['run_id']
        if selected and run_id not in selected:
            continue
        directory = root/run_id
        completed_run(directory, run_id)
        summary = json.loads((directory/policy/'summary.json').read_text())
        if summary.get('run_id') != run_id or summary.get('label') != policy:
            raise ValueError(f'{directory}/{policy}: invalid result identity')
        metadata = dict(row)
        for name in ('locus_manifest', 'loci_file'):
            if metadata.get(name):
                metadata[name] = str(resolve_manifest_path(metadata[name], manifest))
        result.append({**metadata, 'caster_tree': str(directory/'guide.nwk'),
            'filter_decisions_tsv': str(directory/policy/'locus_decisions.tsv'),
            'filtered_caster_tree': str(directory/policy/'caster.nwk'),
            'filtered_loci_file': str(directory/policy/'loci.txt'),
            'n_total': summary['n_total'], 'n_kept': summary['n_kept'], 'n_removed_total': summary['n_removed'],
            'threshold_label': policy, 'threshold_quantile': summary['spec'].get('q', ''),
            'threshold_source': summary['spec']['source'], 'threshold_policy': summary['spec'].get('policy', '')})
    if not result or (selected and selected != {row['run_id'] for row in result}):
        raise ValueError('no or unknown selected one-shot run IDs')
    return result
