"""Run-scoped calibration policies; only reduced results survive each loss block."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
import math

import numpy as np

from .empirical import decimal_probability, empirical_quantile, probability_text
from .input_conditioned_null import policy_sample_indices, sample_gene_tree_panel
from .null_scheduler import null_losses
from .observations import precision12


@dataclass(frozen=True, slots=True)
class ThresholdSpec:
    label: str
    source: str = 'gene-tree-sim'
    policy: str = 'locus'
    q: str = '0.95'
    policy_q: str = '0'

    def validate(self, transform, sample_size):
        q, pq = decimal_probability(self.q), decimal_probability(self.policy_q)
        if self.source not in {'gene-tree-sim','analytic-pb'}:
            raise ValueError('unsupported threshold source')
        if not Decimal(0) < q < Decimal(1) or (self.source=='analytic-pb' and q<=Decimal('0.5')):
            raise ValueError('threshold quantile must be in (0,1); analytic requires (0.5,1)')
        allowed = {'locus','standardized-quantile','standardized-pooled'} if self.source=='gene-tree-sim' else {'locus','standardized-quantile','pooled-tail'}
        if self.policy not in allowed:
            raise ValueError(f'unsupported {self.source}/{self.policy}')
        if self.policy!='standardized-quantile' and pq!=0:
            raise ValueError('policy quantile applies only to standardized-quantile')
        if transform not in {'mass','raw-alt'} or (transform!='raw-alt' and (self.source=='analytic-pb' or self.policy!='locus')):
            raise ValueError(f'unsupported {transform}/{self.source}/{self.policy}')
        if sample_size < 0 or (self.policy=='locus' and sample_size):
            raise ValueError('policy sample size requires a global calibration policy')


def analytic_pooled_tail(distributions, target):
    """Historical mean-tail bisection, unchanged probability-mass semantics."""
    variable=[d for d in distributions if d.summary.standard_deviation>0]
    if not variable:
        return math.nan
    def average(z):
        return float(np.mean([d.survival(d.summary.mean+z*d.summary.standard_deviation) for d in variable]))
    lower,upper=0.,1.
    while average(lower)<=target:
        lower-=1
        if lower < -100: raise ArithmeticError('could not bracket pooled-tail threshold')
    while average(upper)>target:
        upper*=2
        if upper > 1e6: raise ArithmeticError('could not bracket pooled-tail threshold')
    for _ in range(100):
        midpoint=(lower+upper)/2
        if average(midpoint)>target: lower=midpoint
        else: upper=midpoint
        if upper-lower<=1e-12*max(1.,abs(midpoint)): break
    return upper


def calibrate(observations, specs, bank, *, sample_size, sample_seed, panel_seed,
              panel_size, reservoir, binary, guide, statistics_seed, threads,
              memory_bytes, work, progress=None):
    loci=observations.loci
    selections={}
    for spec in specs:
        key=(spec.source,spec.policy)
        if key in selections: continue
        eligible=[l.index for l in loci if l.sd>0 and (spec.policy!='standardized-pooled' or l.status=='eligible')]
        if spec.policy=='standardized-pooled' and sample_size>len(eligible):
            raise ValueError(f'pooled policy requested {sample_size} loci but only {len(eligible)} eligible')
        selected=list(range(len(loci))) if spec.policy=='locus' else policy_sample_indices(eligible,sample_size,sample_seed)
        if not selected and spec.source=='gene-tree-sim':
            raise ValueError(f'{spec.policy}: no eligible calibration loci')
        selections[key]=selected
    requested={}
    panels={}
    pool=None
    pool_indices={}
    effective_reservoir,effective_panel=reservoir,panel_size
    if ('gene-tree-sim','standardized-pooled') in selections:
        if bank.count<reservoir and sample_size>0:
            effective_reservoir=bank.count
            effective_panel=min(panel_size,effective_reservoir)
        if effective_reservoir>bank.count or effective_panel>effective_reservoir:
            raise ValueError('panel/reservoir exceeds simulation bank')
        selected=selections['gene-tree-sim','standardized-pooled']
        if len(selected)*effective_panel*12 > memory_bytes//3:
            raise MemoryError('pooled loss and panel arrays exceed internal memory budget')
        pool=np.full((len(selected),effective_panel),np.nan)
        for rank,index in enumerate(selected):
            ids=sample_gene_tree_panel(locus_index=index,panel_seed_base=panel_seed,
                                      reservoir_size=effective_reservoir,panel_size=effective_panel).astype(np.uint32)
            requested[index]=panels[index]=ids
            pool_indices[index]=rank
    full_indices=set()
    for (source,policy),selected in selections.items():
        if source=='gene-tree-sim' and policy!='standardized-pooled':
            full_indices.update(selected)
    if full_indices:
        full=np.arange(bank.count,dtype=np.uint32)
        for index in full_indices:
            if loci[index].denominator:
                requested[index]=full
    qs=sorted({s.q for s in specs if s.source=='gene-tree-sim' and s.policy!='standardized-pooled'})
    # At most q_count * locus_count small reduction records, never N*G losses.
    reduced={q:{} for q in qs}
    def consume(index,ids,values):
        locus=loci[index]
        if index in panels:
            positions=np.searchsorted(ids,panels[index])
            pool[pool_indices[index]]=(values[positions]-locus.mean)/locus.sd
        if index in full_indices:
            for q in qs:
                result=empirical_quantile(values,q)
                reduced[q][index]=asdict(result)
    # Observation records remain resident while pooled panels and native workers
    # are active. Reserve those bytes before deriving the worker budget so the
    # public memory limit covers the combined one-shot operation.
    observation_bytes = sum(
        locus.weights.nbytes + locus.split_ids.nbytes + len(locus.present) + 2048
        for locus in loci
    )
    retained_bytes = observation_bytes + (pool.nbytes if pool is not None else 0)
    scheduler_memory = memory_bytes - retained_bytes - (16 << 20)
    if requested and scheduler_memory < (32 << 20):
        raise MemoryError(
            'insufficient memory remains for native null workers after retained '
            'observations and pooled panels; increase --memory-budget-gib'
        )
    scheduling = null_losses(observations,requested,bank,binary=binary,guide=guide,seed=statistics_seed,
                             threads=threads,memory_bytes=scheduler_memory,work=work,consume=consume,
                             progress=progress) if requested else {}
    outputs={}
    for spec in specs:
        selected=selections[spec.source,spec.policy]
        selected_rank={index:rank+1 for rank,index in enumerate(selected)}
        z=math.nan
        empirical=None
        thresholds=np.full(len(loci),np.nan)
        candidates={}
        analytic_meta={}
        if spec.source=='analytic-pb':
            distributions={i:loci[i].distribution(observations.probabilities) for i in selected}
            for i,d in distributions.items():
                result=d.quantile(float(spec.q))
                analytic_meta[i]=asdict(result)
                thresholds[i]=result.threshold
                if loci[i].sd>0:
                    candidates[i]=(result.threshold-loci[i].mean)/loci[i].sd
            if spec.policy=='pooled-tail':
                z=analytic_pooled_tail(list(distributions.values()),1.0-float(spec.q))
        elif spec.policy=='standardized-pooled':
            empirical=asdict(empirical_quantile(pool.reshape(-1),spec.q))
            z=empirical['threshold']
        else:
            for i in selected:
                if i in reduced[spec.q]:
                    thresholds[i]=precision12(reduced[spec.q][i]['threshold'])
                    if loci[i].sd>0: candidates[i]=(thresholds[i]-loci[i].mean)/loci[i].sd
        if spec.policy=='standardized-quantile':
            finite=[candidates[i] for i in selected if i in candidates and math.isfinite(candidates[i])]
            if finite:
                empirical=asdict(empirical_quantile(finite,spec.policy_q))
                z=empirical['threshold']
            elif spec.source=='gene-tree-sim':
                raise ValueError('sampled simulation policy produced no finite candidate z values')
        if spec.policy!='locus':
            for locus in loci:
                thresholds[locus.index]=locus.mean+z*locus.sd if locus.sd>0 and math.isfinite(z) else locus.mean
                if spec.policy=='standardized-pooled' and locus.status in {'ineligible_no_valid_branches','excluded_nonfinite_observed_score'}:
                    thresholds[locus.index]=math.nan
        rows=[]
        for locus in loci:
            i=locus.index
            local=(reduced.get(spec.q,{}).get(i) if spec.source=='gene-tree-sim' and spec.policy!='standardized-pooled' else None)
            rows.append(dict(locus_index=i,locus_id=str(i),alignment_path=str(locus.path),
                threshold_score=precision12(thresholds[i]),threshold_source=spec.source,
                threshold_policy=spec.policy,threshold_quantile=spec.q,threshold_label=spec.label,
                policy_quantile=spec.policy_q,policy_z=z,policy_sampled=int(i in selected_rank),
                policy_sample_rank=selected_rank.get(i,''),policy_sample_size_requested=sample_size,
                policy_sample_size_used=len(selected),policy_sample_seed=sample_seed,
                analytic_mean=locus.mean,analytic_variance_independent=locus.variance,analytic_sd_independent=locus.sd,
                calibration_status=locus.status,candidate_threshold_z=candidates.get(i,math.nan),
                empirical_algorithm=local['algorithm'] if local else '',
                empirical_q=local['q'] if local else '',empirical_n=local['n'] if local else '',
                empirical_rank=local['rank'] if local else '',empirical_ties=local['ties'] if local else '',
                empirical_strict_exceedance_count=local['strict_exceedance_count'] if local else '',
                analytic_method=analytic_meta.get(i,{}).get('method',''),
                analytic_tail_probability=analytic_meta.get(i,{}).get('tail_probability','')))
        outputs[spec.label]=dict(rows=rows,global_empirical=empirical,policy_z=z,spec=asdict(spec),
                                 panel_size=effective_panel if spec.policy=='standardized-pooled' else None,
                                 reservoir=effective_reservoir if spec.policy=='standardized-pooled' else None,
                                 selected_indices=selected)
    return outputs,dict(scheduling=scheduling,retained_observation_bytes=observation_bytes,
                        retained_pool_bytes=pool.nbytes if pool is not None else 0,
                        scheduler_memory_bytes=scheduler_memory if requested else 0,
                        gene_id_start=0,gene_id_stop=bank.count if bank else 0,
                        consumed_holdout=bool(full_indices),panel_identity='iils-panel-seed-v2',
                        panel_sampling='splitmix64-rejection-v1')
