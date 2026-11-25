"""Bounded leaf-set requests, sharing each gene statistic within a locus block."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .native_worker import NativeWorker, ProtocolError


def null_losses(observations, ids_by_locus, bank, *, binary: Path, guide: Path,
                seed: int, threads: int, memory_bytes: int, work: Path, consume, progress=None):
    """Deliver at most one bounded group of full loss arrays per worker at a time.

    consume runs on the caller thread. It must reduce or pool the arrays before
    returning. The queue never contains more requests than live workers.
    """
    groups = {}
    for locus in observations.loci:
        if locus.index in ids_by_locus and locus.denominator:
            groups.setdefault(locus.present, []).append(locus)
    if not groups:
        return dict(workers=0, leaf_sets=0, gene_statistics=0, requests=0)
    worker_count = min(threads, max(1,memory_bytes // (128 << 20)),
                       sum((len(v)+63)//64 for v in groups.values()))
    per_worker = memory_bytes // worker_count
    max_genes = max(len(ids) for ids in ids_by_locus.values())
    if max_genes*8*4 > per_worker:
        raise MemoryError('one full locus loss vector exceeds worker memory budget; increase --memory-budget-gib')
    locus_block = min(64, max(1,per_worker // max(1,max_genes*8*4)))
    # Native ASTER quartet state is much larger than its compact topology input.
    gene_block = min(512,max(1,per_worker // max(1,len(bank.taxa)*4096)))
    def packed_batches():
        packed, locus_count, gene_count = [], 0, 0
        for mask, loci in groups.items():
            for start in range(0, len(loci), locus_block):
                block = loci[start:start+locus_block]
                union = np.unique(np.concatenate([ids_by_locus[l.index] for l in block]))
                if packed and (locus_count+len(block)>locus_block or gene_count+len(union)>gene_block):
                    yield packed
                    packed, locus_count, gene_count = [], 0, 0
                packed.append((mask, block, union))
                locus_count += len(block)
                gene_count += len(union)
        if packed:
            yield packed
    batches = iter(packed_batches())
    counters = dict(workers=worker_count,leaf_sets=len(groups),gene_statistics=0,requests=0,
                    gene_block_size=gene_block,locus_block_size=locus_block)

    def process(worker,batch):
        losses = {l.index: np.full(len(ids_by_locus[l.index]),np.nan) for _,loci,_ in batch for l in loci}
        contexts = [([i for i in range(len(bank.taxa)) if (int.from_bytes(mask,'little')>>i)&1], loci, union)
                    for mask,loci,union in batch]
        requests = 0
        for start in range(0,max(len(union) for _,_,union in contexts),gene_block):
            jobs,positions = [],{}
            for group_id,(present,loci,union) in enumerate(contexts):
                genes = union[start:start+gene_block]
                if not len(genes):
                    continue
                targets=[]
                for locus in loci:
                    ids = ids_by_locus[locus.index]
                    a,b = np.searchsorted(ids,[genes[0],genes[-1]],side='left')
                    b += int(b<len(ids) and ids[b]==genes[-1])
                    if a==b:
                        continue
                    targets.append((locus,ids[a:b]))
                    positions[locus.index]=(a,b)
                jobs.append((group_id,present,targets))
            values = worker.losses(jobs,observations.split_keys)
            requests += 1
            for index,ids,array in values:
                a,b = positions[index]
                if np.any(np.isfinite(losses[index][a:b])):
                    raise ProtocolError('duplicate loss block')
                losses[index][a:b]=array
        if any(not np.all(np.isfinite(v)) for v in losses.values()):
            raise ProtocolError('incomplete loss block')
        return losses,sum(len(union) for _,_,union in contexts),requests

    with ExitStack() as stack:
        workers = []
        for index in range(worker_count):
            worker = stack.enter_context(NativeWorker(binary,guide,threads=1,seed=seed,
                         log=work/f'null-worker-{index}.log',max_frame=min(1<<30,per_worker//4)))
            worker.open_bank(bank)
            workers.append(worker)
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=worker_count))
        pending = {}
        for worker in workers:
            batch = next(batches,None)
            if batch is not None:
                pending[executor.submit(process,worker,batch)] = worker
        try:
            while pending:
                done,_ = wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    worker = pending.pop(future)
                    losses,count,requests = future.result()
                    counters['gene_statistics'] += count
                    counters['requests'] += requests
                    if progress is not None:
                        progress(dict(counters))
                    for index,array in losses.items():
                        consume(index,ids_by_locus[index],array)
                    del losses
                    batch = next(batches,None)
                    if batch is not None:
                        pending[executor.submit(process,worker,batch)] = worker
        except BaseException:
            for worker in workers:
                if worker.process.poll() is None:
                    worker.process.terminate()
            for future in pending:
                future.cancel()
            raise
    return counters
