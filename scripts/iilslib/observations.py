"""Compact observations with the historical numeric serialization boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np

from .gene_tree_null import species_penalty
from .native_worker import NativeWorker
from .statistics import caster_support_by_split, _sequence_conflict
from .weighted_poisson_binomial import WeightedBernoulliDistribution


def precision12(value: float) -> float:
    return float(f'{value:.12g}')


@dataclass(slots=True)
class Observation:
    index: int
    path: Path
    sha256: str
    input_bytes: int
    present: bytes
    split_ids: np.ndarray
    weights: np.ndarray  # unnormalized sum(transform_weight * penalty) by split
    denominator: int
    scores: dict
    mean: float
    variance: float
    sd: float
    status: str

    def distribution(self, probabilities):
        return WeightedBernoulliDistribution(
            self.weights / self.denominator if self.denominator else self.weights,
            probabilities[self.split_ids])


@dataclass(slots=True)
class Observations:
    loci: list[Observation]
    taxa: tuple[str, ...]
    split_keys: tuple[str, ...]
    probabilities: np.ndarray


def inventory(loci_file: Path) -> list[Path]:
    # Explicit filtering inputs resolve from the project root, as before.
    from .paths import resolve_project_path
    paths = [resolve_project_path(line.strip()).resolve()
             for line in loci_file.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('loci inventory must be nonempty and contain unique paths')
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def observe(paths: list[Path], guide: Path, worker: NativeWorker, *, transform: str,
            batch_size: int, memory_bytes: int, diagnostic=None, progress=None) -> Observations:
    if batch_size < 1 or memory_bytes < 1:
        raise ValueError('positive batch size and memory budget required')
    support = caster_support_by_split(guide)
    splits = tuple(sorted(support))
    split_index = {key: i for i, key in enumerate(splits)}
    penalties = np.asarray([species_penalty({f'species_p{i+1}': f'{v:.12g}'
                                            for i,v in enumerate(support[key].proportions)})
                            for key in splits])
    probabilities = np.asarray([(2.0/3.0)*math.exp(-p) for p in penalties])
    result, taxa = [], None
    # Native state can substantially exceed FASTA size; also bound wire frames.
    max_input = min(memory_bytes // 16, worker.max_frame // 4)
    pending, pending_bytes = [], 0

    def consume(entries):
        nonlocal taxa
        names, metadata, keys, raw = worker.msa(entries)
        if taxa is None:
            taxa = names
        # Native workers may choose a different stable branch ordering per
        # request; raw rows and keys are positional within each response, so
        # verify the identity set while retaining that response-local order.
        if names != taxa or set(keys) != set(splits):
            raise ValueError('guide taxon/split identity changed between MSA batches')
        for column, (index, n_raw, n_used, n_blocks, present) in enumerate(metadata):
            _, path, content = entries[column]
            if len(present) == 0:
                raise ValueError(f'alignment has no usable FASTA taxa: {path}')
            preliminary = len(present) < 4 or n_blocks == 0 or (transform in {'gap','winner'} and n_used == 0)
            valid = 0
            skipped = len(keys) if preliminary else 0
            total = weighted = ratios = confidence = 0.0
            alt = 0
            ids, weights = [], []
            for branch, key in enumerate(keys):
                if preliminary:
                    continue
                values = raw[branch,column]
                scores = tuple(values['scores'][:3])
                capacity = int(values['capacity'])
                bs = support[key]
                _, evidence, conflict, has_score = _sequence_conflict(scores,capacity,transform,1e-12)
                if has_score:
                    loss = conflict * evidence * bs.penalty if transform == 'mass' else conflict * bs.penalty
                    valid += 1
                    total += loss
                    weighted += bs.confidence * loss
                    ratios += bs.ils_ratio
                    confidence += bs.confidence
                    alt += int(max(scores[1:]) > scores[0])
                else:
                    skipped += 1
                if capacity > 0:
                    sid = split_index[key]
                    factor = (precision12(evidence) if transform == 'mass' else
                              max(0.,float(values['scores'][3])) / capacity)
                    ids.append(sid)
                    weights.append(factor * penalties[sid])
                if diagnostic is not None:
                    diagnostic(index,key,scores,float(values['scores'][3]),capacity,evidence)
            order = np.argsort(ids)
            ids = np.asarray(ids,dtype=np.uint32)[order]
            weights = np.asarray(weights,dtype=np.float64)[order]
            denominator = len(ids)
            dist = WeightedBernoulliDistribution(weights / denominator if denominator else weights,
                                                 probabilities[ids])
            summary = dist.summary
            scores = dict(n_taxa_present=len(present), n_sites_raw=n_raw,n_sites_used=n_used,
                          n_internal_gene_blocks=n_blocks,n_valid_branches=valid,n_skipped_branches=skipped,
                          loss_norm_sum=precision12(total),loss_norm_mean=precision12(total/valid) if valid else math.nan,
                          loss_weighted_norm_sum=precision12(weighted),
                          loss_weighted_norm_mean=precision12(weighted/valid) if valid else math.nan,
                          alt_winning_branch_fraction=precision12(alt/valid) if valid else math.nan,
                          mean_ils_ratio=precision12(ratios/valid) if valid else math.nan,
                          mean_branch_confidence=precision12(confidence/valid) if valid else math.nan,
                          support_source='caster-annotated',score_transform=transform)
            mask = np.zeros((len(names)+63)//64,dtype='<u8')
            for taxon in present:
                mask[int(taxon)//64] |= np.uint64(1) << np.uint64(int(taxon)%64)
            status = ('ineligible_no_valid_branches' if not denominator else
                      'degenerate_zero_variance' if summary.standard_deviation == 0 else 'eligible')
            if not math.isfinite(scores['loss_norm_mean']):
                status = 'excluded_nonfinite_observed_score'
            result.append(Observation(index,path,hashlib.sha256(content).hexdigest(),len(content),
                                      mask.tobytes(),ids,weights,denominator,scores,summary.mean,
                                      summary.variance,summary.standard_deviation,status))
            if progress is not None:
                progress(len(result), len(paths))

    retained_bytes = 0
    for index,path in enumerate(paths):
        size = path.stat().st_size
        if size > max_input:
            raise ValueError(f'{path}: single alignment exceeds MSA byte budget ({size} > {max_input})')
        if pending and (len(pending) >= batch_size or pending_bytes + size > max_input):
            consume(pending)
            retained_bytes += sum(o.weights.nbytes+o.split_ids.nbytes+len(o.present)+2048
                                  for o in result[-len(pending):])
            if retained_bytes > memory_bytes // 2:
                raise MemoryError('compact observations exceed half of the internal memory budget')
            pending, pending_bytes = [], 0
        content = path.read_bytes()
        if len(content) > max_input:
            raise ValueError(f'{path}: alignment grew beyond the MSA byte budget')
        pending.append((index,path,content))
        pending_bytes += len(content)
    if pending:
        consume(pending)
        retained_bytes += sum(o.weights.nbytes+o.split_ids.nbytes+len(o.present)+2048
                              for o in result[-len(pending):])
        if retained_bytes > memory_bytes // 2:
            raise MemoryError('compact observations exceed half of the internal memory budget')
    return Observations(result,taxa,splits,probabilities)
