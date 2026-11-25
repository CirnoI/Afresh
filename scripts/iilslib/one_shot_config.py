"""Explicit one-shot inputs and lossless decimal threshold configuration."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import sys

from .empirical import probability_text
from .observations import inventory
from .caster import validate_name_map
from .one_shot_calibration import ThresholdSpec
from .paths import resolve_project_path
from .randomness import add_seed_argument, effective_seed, resolve_seeds
from .tsv import read_tsv

RETIRED={'--cache-root','--simulation-cache-root','--simulation-manifest','--discordance-manifest',
         '--threshold-manifest','--stage','--start-at','--stop-after','--force','--rebuild-support','--prune-workers'}


def label(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value) or value in {'.','..'}:
        raise ValueError(f'unsafe or empty result label: {value!r}')
    return value


def quantile_label(q):
    text=probability_text(q)
    return 'q'+(text.split('.',1)[1] if '.' in text else text)


def parse_args(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    obsolete={arg.split('=',1)[0] for arg in argv}&RETIRED
    if obsolete:
        raise ValueError('retired one-shot options: '+', '.join(sorted(obsolete))+
                         '; use a new --out-root and --scratch-root; internal stages and cache recovery were removed')
    p=argparse.ArgumentParser(description='Run iILS once, using bounded MSA batches and a private indexed MSC bank. Failure requires a fresh run.')
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--loci-file',type=Path)
    inputs.add_argument('--run-manifest',type=Path,help='TSV with run_id,loci_file and optional guide_tree,constraint_tree,support_loci_file,name_map,caster_seed; no output paths.')
    guides=p.add_mutually_exclusive_group()
    guides.add_argument('--guide-tree',type=Path)
    guides.add_argument('--constraint-tree',type=Path)
    p.add_argument('--support-loci-file',type=Path)
    p.add_argument('--name-map','--namemap','--mapping',dest='name_map',type=Path,
                   help='Two-column sequence/alignment label to guide-tree species label map.')
    p.add_argument('--run-name',default='run')
    p.add_argument('--run-id',action='append',dest='run_ids')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--scratch-root',type=Path,required=True,help='Private work is created below this path; local SSD recommended.')
    p.add_argument('--status-root',type=Path,
                   help='Durable directory for live per-run status JSON; defaults beside a direct result or under a batch result root.')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-budget-gib',type=float,default=4,help='Internal arrays/batches/work queues; does not cap external CASTER RSS.')
    p.add_argument('--msa-batch-size',type=int,default=256)
    p.add_argument('--aster-dir',type=Path,default=Path('ASTER'))
    p.add_argument('--statistics-bin',type=Path,default=Path('Afresh/bin/iils-statistics'))
    p.add_argument('--build-statistics',action='store_true')
    p.add_argument('--n-gene-trees',type=int,default=50000)
    p.add_argument('--max-caster-cu-length',type=float,default=20.)
    add_seed_argument(p)
    for name in ('simulation-seed','statistics-seed','caster-seed','filtered-caster-seed','policy-sample-seed','gene-tree-panel-seed'):
        p.add_argument('--'+name,type=int)
    p.add_argument('--score-transform',choices=['mass','raw-alt','gap','winner'],default='raw-alt')
    p.add_argument('--score-column',choices=['loss_norm_mean','loss_norm_sum','loss_weighted_norm_mean','loss_weighted_norm_sum'],default='loss_norm_mean')
    p.add_argument('--filter-mode',choices=['threshold','quantile'],default='threshold',help='quantile preserves the historical fixed drop-fraction count and stable tie ordering.')
    p.add_argument('--drop-fraction',type=float,default=.05)
    p.add_argument('--threshold-source',choices=['gene-tree-sim','analytic-pb'],default='gene-tree-sim')
    p.add_argument('--threshold-policy',choices=['locus','standardized-quantile','standardized-pooled','pooled-tail'],default='locus')
    p.add_argument('--threshold-quantile',action='append',dest='threshold_quantiles',help='Decimal text; repeat for shared null computation. Default: 0.95.')
    p.add_argument('--threshold-label',action='append',dest='threshold_labels')
    p.add_argument('--policy-quantile',action='append',dest='policy_quantiles',help='Decimal text; repeat for shared low5/median results. Default: 0.')
    p.add_argument('--threshold-specs',type=Path,help='JSON list of {label,source,policy,q,policy_q}; all q fields must be decimal strings.')
    p.add_argument('--threshold-spec',action='append',dest='threshold_records',help='One JSON {label,source,policy,q,policy_q}; repeat to share compatible strategies without an intermediate file. q fields must be decimal strings.')
    p.add_argument('--policy-sample-size',type=int,default=0)
    p.add_argument('--gene-tree-panel-size',type=int,default=100)
    p.add_argument('--calibration-reservoir-size',type=int,default=40000)
    p.add_argument('--build-filtered-tree',action='store_true')
    p.add_argument('--keep-diagnostics',action='store_true',help='Retain diagnostics/msa_branches.tsv and threshold_details.json; MSC bank and discordance tables are never published.')
    p.add_argument('--keep-failed-temp',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p.parse_args(argv)


def normalize(args):
    resolve_seeds(args,'simulation_seed','statistics_seed','filtered_caster_seed','policy_sample_seed','gene_tree_panel_seed')
    if args.caster_seed is not None: effective_seed(args,'caster_seed')
    if args.threads<1 or args.msa_batch_size<1 or not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib<=0:
        raise ValueError('threads, batch size and memory budget must be positive')
    if not 1<=args.n_gene_trees<=2147483647 or not math.isfinite(args.max_caster_cu_length) or args.max_caster_cu_length<=0:
        raise ValueError('invalid simulation dimensions or CU length')
    if args.gene_tree_panel_size<1 or args.calibration_reservoir_size<1 or args.policy_sample_size<0:
        raise ValueError('invalid panel/reservoir/policy sample size')
    if not 0<=args.drop_fraction<1:
        raise ValueError('drop fraction must be in [0,1)')
    for name in ('out_root','scratch_root','status_root','aster_dir','statistics_bin','loci_file','run_manifest','guide_tree',
                 'constraint_tree','support_loci_file','name_map','threshold_specs'):
        value=getattr(args,name)
        if value is not None: setattr(args,name,resolve_project_path(value).resolve())
    if (args.out_root == args.scratch_root
            or args.out_root.is_relative_to(args.scratch_root)
            or args.scratch_root.is_relative_to(args.out_root)):
        raise ValueError('--out-root and --scratch-root must be separate directory trees')
    if args.filter_mode=='quantile':
        specs=[]
    elif args.threshold_specs or args.threshold_records:
        if args.threshold_specs and args.threshold_records:
            raise ValueError('use either --threshold-specs or repeated --threshold-spec')
        if args.threshold_quantiles or args.policy_quantiles or args.threshold_labels:
            raise ValueError('explicit threshold specs cannot be combined with threshold/policy quantile or label options')
        records=(json.loads(args.threshold_specs.read_text()) if args.threshold_specs
                 else [json.loads(record) for record in args.threshold_records])
        if not isinstance(records,list) or not records: raise ValueError('threshold specs must be a nonempty JSON list')
        specs=[ThresholdSpec(**record) for record in records]
    else:
        qs=[probability_text(q) for q in args.threshold_quantiles or ['0.95']]
        pqs=[probability_text(q) for q in args.policy_quantiles or ['0']]
        combinations=[(q,pq) for q in qs for pq in pqs]
        if args.threshold_labels and len(args.threshold_labels)!=len(combinations):
            raise ValueError('threshold labels must match the q × policy-q result count')
        names=args.threshold_labels or [quantile_label(q)+(('_z'+quantile_label(pq)) if len(pqs)>1 else '') for q,pq in combinations]
        specs=[ThresholdSpec(name,args.threshold_source,args.threshold_policy,q,pq) for name,(q,pq) in zip(names,combinations)]
    specs=[replace(spec,q=probability_text(spec.q),policy_q=probability_text(spec.policy_q)) for spec in specs]
    if len({label(s.label) for s in specs})!=len(specs): raise ValueError('duplicate threshold labels')
    for spec in specs:
        spec.validate(args.score_transform,args.policy_sample_size)
        if spec.policy=='standardized-pooled':
            reservoir=min(args.n_gene_trees,args.calibration_reservoir_size) if args.policy_sample_size else args.calibration_reservoir_size
            panel=min(args.gene_tree_panel_size,reservoir) if args.policy_sample_size and args.n_gene_trees<args.calibration_reservoir_size else args.gene_tree_panel_size
            if reservoir>args.n_gene_trees or panel>reservoir: raise ValueError('pooled panel/reservoir exceeds bank')
    if specs and args.score_column!='loss_norm_mean':
        raise ValueError('calibrated thresholds require score-column loss_norm_mean')
    return specs


@dataclass(slots=True)
class RunInput:
    run_id: str
    loci_file: Path
    paths: list[Path]
    guide_tree: Path | None
    constraint_tree: Path | None
    support_loci_file: Path | None
    name_map: Path | None
    caster_seed: int
    output: Path
    metadata: dict


def inputs(args):
    if args.out_root.exists():
        raise FileExistsError(f'result already exists; choose a fresh output: {args.out_root}')
    if args.loci_file:
        rows=[dict(run_id=args.run_name,loci_file=str(args.loci_file),
                   **{k:str(getattr(args,k)) if getattr(args,k) else '' for k in ('guide_tree','constraint_tree','support_loci_file','name_map')})]
    else:
        if any((args.guide_tree,args.constraint_tree,args.support_loci_file,args.name_map)):
            raise ValueError('batch input paths must be provided in input manifest columns')
        rows=read_tsv(args.run_manifest,required={'run_id','loci_file'})
        retired={'out_dir','caster_tree','iils_scores_tsv','iils_branch_output_tsv'}
        if rows and retired.intersection(rows[0]):
            raise ValueError('old run manifest schema: replace caster_tree with guide_tree only for an existing annotated guide, and remove output/score columns; use support_loci_file/constraint_tree for inference')
    ids=[label(row['run_id']) for row in rows]
    if len(set(ids))!=len(ids): raise ValueError('duplicate run IDs')
    if args.run_ids and not set(args.run_ids)<=set(ids): raise ValueError('unknown selected run ID')
    result=[]
    for row in rows:
        if args.run_ids and row['run_id'] not in args.run_ids: continue
        paths={name:resolve_project_path(row[name]).resolve() if row.get(name) else None
               for name in ('loci_file','guide_tree','constraint_tree','support_loci_file')}
        name_map_value = row.get('name_map') or row.get('namemap') or row.get('mapping')
        paths['name_map'] = resolve_project_path(name_map_value).resolve() if name_map_value else None
        if paths['guide_tree'] and paths['constraint_tree']: raise ValueError('guide and constraint are mutually exclusive')
        for path in paths.values():
            if path is not None and not path.is_file(): raise FileNotFoundError(path)
        if paths['name_map']:
            validate_name_map(paths['name_map'])
        if paths['guide_tree']:
            from .caster import has_caster_support_annotation
            if not has_caster_support_annotation(paths['guide_tree']):
                raise ValueError('guide_tree must contain CASTER raw s1/s2/s3 annotations')
        locus_paths=inventory(paths['loci_file'])
        if paths['support_loci_file']: inventory(paths['support_loci_file'])
        output=args.out_root if args.loci_file else args.out_root/row['run_id']
        if output.exists(): raise FileExistsError(f'result already exists; choose a fresh output: {output}')
        # Output must never own an input path; this also protects experiment preparation roots.
        for path in [*locus_paths,*[p for p in paths.values() if p]]:
            if path.is_relative_to(output): raise ValueError('result directory contains an input')
        result.append(RunInput(row['run_id'],paths['loci_file'],locus_paths,paths['guide_tree'],paths['constraint_tree'],
                               paths['support_loci_file'],paths['name_map'],effective_seed(args,'caster_seed',row.get('caster_seed')),
                               output,{k:v for k,v in row.items() if k not in paths and k not in {'namemap','mapping'}}))
    if not result: raise ValueError('no selected runs')
    return result
