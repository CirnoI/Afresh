"""Run-private indexed topology bank, preserving the sequential DendroPy stream."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import struct

from .gene_tree_null import load_cu_species_tree

MAGIC = b'IILSBK01'
HEADER = struct.Struct('<8sII')
NODE = struct.Struct('<id')  # leaf taxon ID or negative child count, branch length


@dataclass(frozen=True, slots=True)
class TreeBank:
    path: Path
    count: int
    taxa: tuple[str, ...]
    bytes: int
    sha256: str


def write_bank(path: Path, trees, count: int, taxa: tuple[str, ...], progress=None) -> TreeBank:
    """Write postorder trees one at a time; the only resident index is G+1 uint64s."""
    if count < 1 or count > 2147483647 or len(set(taxa)) != len(taxa):
        raise ValueError('invalid topology bank dimensions')
    ids = {name: i for i, name in enumerate(taxa)}
    offsets = []
    with path.open('xb') as handle:
        handle.write(HEADER.pack(MAGIC, count, len(taxa)))
        handle.write(bytes(8 * (count + 1)))
        for tree in trees:
            if len(offsets) >= count:
                raise ValueError('too many simulated trees')
            offsets.append(handle.tell())
            nodes = list(tree.postorder_node_iter())
            handle.write(struct.pack('<I', len(nodes)))
            for node in nodes:
                children = node.child_nodes()
                code = -len(children) if children else ids[node.taxon.label]
                if code == -1:
                    raise ValueError('unary topology node in simulation')
                handle.write(NODE.pack(code, node.edge.length or 0.0))
            if progress is not None and (len(offsets) == 1 or len(offsets) % 512 == 0 or len(offsets) == count):
                progress(len(offsets), count)
        if len(offsets) != count:
            raise ValueError('simulation ended before the requested tree count')
        offsets.append(handle.tell())
        handle.seek(HEADER.size)
        handle.write(struct.pack('<' + 'Q' * len(offsets), *offsets))
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return TreeBank(path, count, taxa, path.stat().st_size, digest.hexdigest())


def simulate_bank(path: Path, guide: Path, taxa: tuple[str, ...], *, count: int,
                  seed: int, max_cu_length: float, progress=None) -> tuple[TreeBank, dict]:
    dendropy, coalescent, namespace, containing, cu_text, stats = load_cu_species_tree(guide,max_cu_length)
    mapping = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
        containing_taxon_namespace=namespace, num_contained=1,
        contained_taxon_label_fn=lambda taxon, _index: taxon.label)
    rng = random.Random(seed)

    def trees():
        for _ in range(count):
            yield coalescent.contained_coalescent_tree(containing_tree=containing,
                gene_to_containing_taxon_map=mapping,edge_pop_size_attr=None,default_pop_size=1,rng=rng)

    bank = write_bank(path,trees(),count,taxa,progress=progress)
    return bank, dict(backend='dendropy',version=dendropy.__version__,seed=seed,
                      species_tree_cu=cu_text,max_cu_length=max_cu_length,
                      n_internal_branches=stats.n_internal_branches)
