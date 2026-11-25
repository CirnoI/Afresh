"""Stage only explicit inputs and publish relocatable results for cluster jobs.

The filtering command remains the common one-shot CLI. This module owns a fresh
local directory; no historical output/cache is copied or used for recovery.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .observations import inventory
from .one_shot import publish, sha256, write_json
from .one_shot_config import inputs, normalize, parse_args
from .tsv import read_tsv, write_tsv


def stage_inputs(runs, work: Path):
    """Copy each distinct file once; the map points back to durable input paths."""
    directory = work / 'inputs'
    directory.mkdir()
    copies = {}
    published_paths = {}
    total_bytes = 0

    def copy_file(source):
        nonlocal total_bytes
        source = Path(source).resolve()
        if source not in copies:
            target = directory / f'{len(copies):08d}{source.suffix}'
            shutil.copyfile(source, target)
            total_bytes += target.stat().st_size
            copies[source] = target
            published_paths[str(target)] = str(source)
        return str(copies[source])

    def copy_list(source, paths):
        source = Path(source).resolve()
        if source not in copies:
            values = [copy_file(path) for path in paths]
            target = directory / f'{len(copies):08d}.txt'
            target.write_text(''.join(value + '\n' for value in values))
            copies[source] = target
            published_paths[str(target)] = str(source)
        return str(copies[source])

    rows = []
    for run in runs:
        row = dict(run_id=run.run_id, loci_file=copy_list(run.loci_file, run.paths),
                   caster_seed=run.caster_seed)
        for name in ('guide_tree', 'constraint_tree', 'name_map'):
            value = getattr(run, name)
            row[name] = copy_file(value) if value else ''
        row['support_loci_file'] = (copy_list(run.support_loci_file, inventory(run.support_loci_file))
                                    if run.support_loci_file else '')
        rows.append(row)
    manifest = work / 'run_manifest.tsv'
    write_tsv(manifest, list(rows[0]), rows)
    return manifest, published_paths, dict(copied_files=len(copies), copied_bytes=total_bytes)


def remap_value(value, mapping):
    if isinstance(value, dict):
        return {key: remap_value(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_value(item, mapping) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def prepare_publication(local: Path, final: Path, destination: Path, mapping: dict):
    """Rewrite actionable paths before committing any result at its durable root."""
    final.mkdir()
    paths = dict(mapping)
    paths[str(local)] = str(destination)
    for source in local.rglob('*'):
        paths[str(source)] = str(destination / source.relative_to(local))
    for source in local.rglob('*'):
        if not source.is_file() or source.name == 'one_shot_receipt.json':
            continue
        relative = source.relative_to(local)
        target = final / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.tsv':
            with source.open(newline='') as handle:
                reader = csv.DictReader(handle, delimiter='\t')
                fields = reader.fieldnames
                write_tsv(target, fields, (remap_value(row, paths) for row in reader))
        elif source.name == 'loci.txt':
            target.write_text(''.join(paths.get(line, line) + '\n' for line in source.read_text().splitlines()))
        elif source.name == 'configuration.json':
            write_json(target, remap_value(json.loads(source.read_text()), paths))
        else:
            shutil.copyfile(source, target)
    receipt_path = local / 'one_shot_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    shutil.copyfile(receipt_path, final / 'local_execution_receipt.json')
    receipt.pop('artifacts', None)
    receipt['local_execution_receipt_sha256'] = sha256(receipt_path)
    return receipt


def execute_local(command, local_root: Path, *, keep_failed=False):
    """Execute a batch command locally, then publish each run on the target FS."""
    args = parse_args(command[2:])
    specs = normalize(args)
    runs = inputs(args)
    if not args.run_manifest:
        raise ValueError('local staging requires an explicit batch run manifest')
    if args.dry_run:
        return subprocess.run(command, check=True)
    destination = args.out_root
    local_root = local_root.resolve()
    if destination == local_root or destination.is_relative_to(local_root) or local_root.is_relative_to(destination):
        raise ValueError('durable output and local staging must be separate directory trees')
    local_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='iils-job-', dir=local_root))
    completed = False
    # Reserve ownership before staging or computation; concurrent jobs cannot
    # both spend hours preparing the same durable destination.
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except BaseException:
        shutil.rmtree(work)
        raise
    states = {run.run_id: 'not_executed' for run in runs}
    try:
        manifest, mapping, staging = stage_inputs(runs, work)
        mapping.update({str(manifest): str(args.run_manifest), str(work/'results'): str(destination),
                        str(work/'scratch'): str(args.scratch_root)})
        config = work / 'thresholds.json'
        write_json(config, [asdict(spec) for spec in specs])
        # The normalized thresholds are embedded in configuration.json, so no
        # external specification file is needed after local execution.
        mapping[str(config)] = None
        # Reconstruct options from the validated namespace so source tree/list
        # options cannot accidentally leak into the staged batch.
        local_command = [command[0], command[1], '--run-manifest', str(manifest),
                         '--out-root', str(work / 'results'), '--scratch-root', str(work / 'scratch'),
                         '--status-root', str(args.status_root or destination / '.status')]
        excluded = {'run_manifest', 'loci_file', 'run_name', 'run_ids', 'out_root', 'scratch_root',
                    'status_root',
                    'guide_tree', 'constraint_tree', 'support_loci_file', 'name_map',
                    'threshold_specs', 'threshold_records', 'threshold_quantiles', 'threshold_labels', 'policy_quantiles'}
        for name, value in vars(args).items():
            if name in excluded or value is None or value is False:
                continue
            local_command.append('--' + name.replace('_', '-'))
            if value is not True:
                local_command.append(str(value))
        if specs:
            local_command.extend(['--threshold-specs', str(config)])
        execution = subprocess.run(local_command)
        for run in runs:
            local = work / 'results' / run.run_id
            if not (local / 'one_shot_receipt.json').is_file():
                continue
            final = work / ('publish-' + run.run_id)
            receipt = prepare_publication(local, final, destination / run.run_id, mapping)
            receipt['staging'] = staging
            publish(final, destination / run.run_id, receipt)
            states[run.run_id] = receipt['status']
        execution.check_returncode()
        if set(states.values()) != {'completed'}:
            raise RuntimeError('local command did not publish all requested runs')
        completed = True
    finally:
        write_json(destination / 'batch_receipt.json', dict(schema='iils-one-shot-batch-v1', runs=states))
        if completed or not keep_failed:
            shutil.rmtree(work)
        else:
            print(f'failed local workspace retained: {work}', flush=True)
