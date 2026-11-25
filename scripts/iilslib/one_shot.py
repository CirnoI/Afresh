"""One execution, private scratch, final-only publication and provenance."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback

from .caster import caster_site_command
from .filtering import ScoreRow,apply_high_loss_filter
from .native_worker import NativeWorker
from .observations import observe,inventory
from .one_shot_calibration import calibrate
from .resources import ResourceMonitor
from .tree_bank import simulate_bank
from .tsv import write_tsv
from .paths import project_root


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk:=handle.read(1<<20): digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value,Path): return str(value)
    if isinstance(value,dict): return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_value(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def write_json(path,value):
    path.write_text(json.dumps(json_value(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class RunProgress:
    """Best-effort live status for operators; it is never a result artifact."""

    def __init__(self, path: Path, *, run_id: str, scratch: Path, output: Path, total_loci: int):
        self.path = path
        self._lock = threading.Lock()
        self._disabled = False
        self._last_write = 0.0
        self._last_print = 0.0
        self._started_monotonic = time.monotonic()
        self._stop = threading.Event()
        self.payload = {
            'schema': 'iils-one-shot-status-v1',
            'run_id': run_id,
            'status': 'running',
            'stage': 'initializing',
            'started_at': self._timestamp(),
            'updated_at': self._timestamp(),
            'elapsed_seconds': 0.0,
            'pid': os.getpid(),
            'host': platform.node(),
            'scratch': str(scratch),
            'output': str(output),
            'total_loci': total_loci,
            'counters': {},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._write()
        except OSError as error:
            self._disabled = True
            print(f'run {run_id}: progress status unavailable: {error}', flush=True)
        self._thread = threading.Thread(target=self._heartbeat, name=f'iils-status-{run_id}', daemon=True)
        self._thread.start()

    @staticmethod
    def _timestamp():
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

    def _write(self):
        if self._disabled:
            return
        now = time.monotonic()
        self.payload['updated_at'] = self._timestamp()
        self.payload['elapsed_seconds'] = now - self._started_monotonic
        temporary = self.path.with_name(f'.{self.path.name}.{os.getpid()}.tmp')
        write_json(temporary, self.payload)
        temporary.replace(self.path)
        self._last_write = now

    def start(self):
        with self._lock:
            try:
                self._write()
            except OSError:
                self._disabled = True
        return self

    def _heartbeat(self):
        while not self._stop.wait(15.0):
            with self._lock:
                try:
                    self._write()
                except OSError:
                    self._disabled = True

    def update(self, *, stage=None, counters=None, force=False, announce=False, **fields):
        with self._lock:
            if stage is not None:
                self.payload['stage'] = stage
            if counters:
                self.payload['counters'].update(counters)
            self.payload.update(fields)
            now = time.monotonic()
            should_write = force or now - self._last_write >= 5.0
            if should_write:
                try:
                    self._write()
                except OSError:
                    self._disabled = True
            if announce or now - self._last_print >= 30.0:
                details = ' '.join(f'{key}={value}' for key, value in self.payload['counters'].items())
                suffix = f' {details}' if details else ''
                print(f"run {self.payload['run_id']}: stage={self.payload['stage']}{suffix}", flush=True)
                self._last_print = now

    def stage(self, name, **fields):
        self.update(stage=name, force=True, announce=True, **fields)

    def finish(self, status, *, error=None):
        self._stop.set()
        self._thread.join(timeout=2)
        with self._lock:
            self.payload['status'] = status
            if error is not None:
                self.payload['error'] = str(error)
            try:
                self._write()
            except OSError:
                self._disabled = True


def table(path,rows):
    if not rows: raise ValueError('cannot infer empty result table schema')
    fields=list(rows[0])
    def formatted():
        for row in rows:
            yield {k:(f'{v:.12g}' if math.isfinite(v) else '') if isinstance(v,float) else v for k,v in row.items()}
    write_tsv(path,fields,formatted())


def code_identity():
    def git(*args):
        return subprocess.check_output(
            ['git', *args], cwd=project_root(), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    try:
        return dict(commit=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')))
    except (OSError, subprocess.CalledProcessError):
        # Slurm wrappers intentionally copy only the runnable project layers to
        # local scratch, so that checkout may not contain .git metadata.
        return dict(commit=None,dirty=None,identity_source='git-unavailable')


def run_caster(binary,loci,output,threads,seed,log,constraint=None,mapping_path=None):
    cmd=caster_site_command(binary,loci,output,threads,seed=seed,constraint_tree=constraint,
                            mapping_path=mapping_path,score_constraint_only=constraint is not None)
    start=time.monotonic()
    with log.open('w') as stream:
        process=subprocess.Popen(list(map(str,cmd)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                 env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try:
            code=process.wait()
            if code: raise RuntimeError(f'CASTER failed with exit {code}; see {log.name}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
            raise
    if not output.is_file() or not output.stat().st_size: raise RuntimeError('CASTER produced no tree')
    return dict(command=list(map(str,cmd)),wall_seconds=time.monotonic()-start,threads=threads,seed=seed)


def status_path(args, run):
    """Choose a durable sidecar path without creating a partial result tree."""
    if args.status_root is not None:
        return args.status_root / f'{run.run_id}.json'
    if args.run_manifest:
        return args.out_root / '.status' / f'{run.run_id}.json'
    return run.output.parent / f'.{run.output.name}.status.json'


def publish(final,output,receipt):
    """Reserve a new result dir and commit the completion receipt last on its FS."""
    output.parent.mkdir(parents=True,exist_ok=True)
    output.mkdir()  # exclusive ownership, even against concurrent publishers
    staging=None
    try:
        staging=Path(tempfile.mkdtemp(prefix='.iils-publish-',dir=output.parent))
        for source in final.rglob('*'):
            relative=source.relative_to(final)
            target=staging/relative
            if source.is_dir(): target.mkdir(parents=True,exist_ok=True)
            elif source.is_file():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                if sha256(source)!=sha256(target): raise OSError(f'publication checksum mismatch: {relative}')
        receipt['artifacts']={str(p.relative_to(staging)):sha256(p) for p in staging.rglob('*') if p.is_file()}
        for source in staging.iterdir(): source.rename(output/source.name)
        write_json(staging/'one_shot_receipt.json',receipt)
        (staging/'one_shot_receipt.json').rename(output/'one_shot_receipt.json')
    except BaseException:
        # The target was reserved by this publisher.  Do not leave an empty or
        # partial result directory that would look like a completed run.
        shutil.rmtree(output, ignore_errors=True)
        raise
    finally:
        if staging is not None: shutil.rmtree(staging)


def execute_run(args,run,specs):
    args.scratch_root.mkdir(parents=True,exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix=f'iils-{run.run_id}-',dir=args.scratch_root))
    final=scratch/'final'
    logs=final/'logs'
    monitor=None
    progress=None
    completed=False
    started=time.monotonic()
    receipt=dict(schema='iils-one-shot-v1',run_id=run.run_id,status='running',scratch=str(scratch),timings={})
    try:
        final.mkdir(); logs.mkdir()
        write_json(scratch/'owner.json',dict(run_id=run.run_id,output=run.output,pid=os.getpid()))
        progress=RunProgress(status_path(args,run),run_id=run.run_id,scratch=scratch,output=run.output,total_loci=len(run.paths)).start()
        receipt['status_path']=str(progress.path)
        progress.stage('initializing')
        monitor=ResourceMonitor(scratch).start()
        receipt.update(code=code_identity(),scratch_free_bytes=shutil.disk_usage(scratch).free,
                       seeds={name:getattr(args,name) for name in ('seed','simulation_seed','statistics_seed','filtered_caster_seed','policy_sample_seed','gene_tree_panel_seed')},
                       binaries={'statistics':sha256(args.statistics_bin)})
        receipt['seeds']['caster_seed']=run.caster_seed
        config={k:v for k,v in vars(args).items() if k not in {'run_ids','status_root'}}
        config['run']={k:v for k,v in asdict(run).items() if k!='paths'}
        config['thresholds']=[asdict(s) for s in specs]
        write_json(final/'configuration.json',config)
        guide=final/'guide.nwk'
        progress.stage('guide')
        if run.guide_tree:
            shutil.copyfile(run.guide_tree,guide)
            receipt['guide_mode']='provided_annotations'
        else:
            support_paths=inventory(run.support_loci_file) if run.support_loci_file else run.paths
            support_list=scratch/'support.txt'
            support_list.write_text(''.join(str(p)+'\n' for p in support_paths))
            receipt['guide_caster']=run_caster(args.aster_dir/'bin/caster-site',support_list,guide,args.threads,
                                              run.caster_seed,logs/'guide-caster.log',run.constraint_tree,run.name_map)
            receipt['guide_mode']='constraint' if run.constraint_tree else 'inferred'
            receipt['support_inputs']=[dict(path=str(p),sha256=sha256(p)) for p in support_paths]
        receipt['guide_sha256']=sha256(guide)
        if run.name_map:
            receipt['name_map_sha256']=sha256(run.name_map)
        if run.constraint_tree: receipt['constraint_sha256']=sha256(run.constraint_tree)
        memory=int(args.memory_budget_gib*(1<<30))
        if memory<32<<20: raise ValueError('internal budget must be at least 32 MiB')
        t=time.monotonic()
        progress.stage('observations')
        with ExitStack() as stack:
            diagnostic=None
            if args.keep_diagnostics:
                import csv
                directory=final/'diagnostics'; directory.mkdir()
                handle=stack.enter_context((directory/'msa_branches.tsv').open('w'))
                writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
                writer.writerow(['locus_index','split_key','score1','score2','score3','score_sum','quartet_capacity','evidence_weight'])
                diagnostic=lambda index,key,scores,total,capacity,evidence: writer.writerow([index,key,*scores,total,capacity,evidence])
            worker=stack.enter_context(NativeWorker(args.statistics_bin,guide,threads=args.threads,seed=args.statistics_seed,
                                      mapping_path=run.name_map,log=logs/'msa.log',max_frame=min(1<<30,memory//4)))
            observations=observe(run.paths,guide,worker,transform=args.score_transform,batch_size=args.msa_batch_size,
                                  memory_bytes=memory,diagnostic=diagnostic,
                                  progress=lambda done,total: progress.update(
                                      counters={'observed_loci': done, 'total_loci': total}))
        receipt['timings']['observations']=time.monotonic()-t
        table(final/'inventory.tsv',[dict(locus_index=l.index,locus_id=str(l.index),alignment_path=str(l.path),
              sha256=l.sha256,bytes=l.input_bytes,present_taxa_mask=l.present.hex()) for l in observations.loci])
        table(final/'scores.tsv',[dict(locus_index=l.index,locus_id=str(l.index),alignment_path=str(l.path),
              calibration_status=l.status,**l.scores) for l in observations.loci])
        receipt.update(locus_count=len(observations.loci),leaf_set_count=len({l.present for l in observations.loci}),
                       taxa=observations.taxa,split_count=len(observations.split_keys),
                       input_fasta_bytes=sum(l.input_bytes for l in observations.loci))
        bank=None
        if any(s.source=='gene-tree-sim' for s in specs):
            t=time.monotonic()
            progress.stage('simulation')
            bank,simulation=simulate_bank(scratch/'bank.bin',guide,observations.taxa,count=args.n_gene_trees,
                                          seed=args.simulation_seed,max_cu_length=args.max_caster_cu_length,
                                          progress=lambda done,total: progress.update(
                                              counters={'simulated_gene_trees': done, 'total_gene_trees': total}))
            receipt['simulation']={**simulation,'bank_bytes':bank.bytes,'bank_sha256':bank.sha256}
            receipt['timings']['simulation']=time.monotonic()-t
        t=time.monotonic()
        results={}
        if specs:
            progress.stage('calibration')
            results,calibration=calibrate(observations,specs,bank,sample_size=args.policy_sample_size,sample_seed=args.policy_sample_seed,
                panel_seed=args.gene_tree_panel_seed,panel_size=args.gene_tree_panel_size,reservoir=args.calibration_reservoir_size,
                binary=args.statistics_bin,guide=guide,statistics_seed=args.statistics_seed,threads=args.threads,
                memory_bytes=memory,work=logs,
                progress=lambda counters: progress.update(counters=counters))
            receipt['calibration']=calibration
        else:
            results['fixed-fraction']=dict(rows=None,spec={'label':'fixed-fraction','source':'fixed-fraction'},global_empirical=None)
        receipt['timings']['calibration']=time.monotonic()-t
        progress.stage('filtering')
        result_rows=[]
        for result_index,(name,result) in enumerate(results.items(),1):
            progress.update(counters={'result_index': result_index, 'result_count': len(results)}, announce=True)
            directory=final/name; directory.mkdir()
            rows=[ScoreRow(l.index,str(l.index),str(l.path),f'{l.scores[args.score_column]:.12g}',
                           l.scores[args.score_column],math.isfinite(l.scores[args.score_column]),score_transform=args.score_transform)
                  for l in observations.loci]
            if not specs:
                apply_high_loss_filter(rows,args.drop_fraction)
            else:
                for score,threshold in zip(rows,result['rows'],strict=True):
                    score.reason='at_or_below_locus_threshold'
                    if not score.is_finite: score.decision='remove'; score.reason='nonfinite_score'
                    elif threshold['calibration_status'] == 'ineligible_no_valid_branches':
                        score.decision='remove'; score.reason='ineligible_no_valid_branches'
                    elif threshold['calibration_status'] == 'excluded_nonfinite_observed_score':
                        score.decision='remove'; score.reason='nonfinite_score'
                    elif score.score>threshold['threshold_score']:
                        score.decision='remove'; score.reason='above_locus_threshold'
                threshold_rows = [dict(run_id=run.run_id, **row) for row in result['rows']]
                table(directory/'thresholds.tsv',threshold_rows)
            decisions=[]
            for i,score in enumerate(rows):
                extra=threshold_rows[i] if result['rows'] else {}
                decisions.append(dict(**extra,**{k:v for k,v in dict(locus_id=score.locus_id,locus_index=score.order,
                    alignment_path=score.alignment_path,decision=score.decision,reason=score.reason,
                    is_finite=int(score.is_finite),score_transform=args.score_transform,**{args.score_column:score.score}).items() if k not in extra}))
            table(directory/'locus_decisions.tsv',decisions)
            kept=[score.alignment_path for score in rows if score.decision=='keep']
            (directory/'loci.txt').write_text(''.join(p+'\n' for p in kept))
            summary=dict(run_id=run.run_id,label=name,n_total=len(rows),n_kept=len(kept),n_removed=len(rows)-len(kept),
                         spec=result['spec'],global_empirical=result['global_empirical'],
                         strict_comparison='score > threshold',policy_z=result.get('policy_z'))
            if args.build_filtered_tree:
                if kept:
                    progress.stage('filtered_tree', result_label=name)
                    summary['caster']=run_caster(args.aster_dir/'bin/caster-site',directory/'loci.txt',directory/'caster.nwk',
                                                args.threads,args.filtered_caster_seed,logs/f'{name}-caster.log',mapping_path=run.name_map)
                else: summary['caster_status']='no_retained_loci'
            write_json(directory/'summary.json',summary)
            result_rows.append(dict(run_id=run.run_id,label=name,n_total=len(rows),n_kept=len(kept),
                n_removed=len(rows)-len(kept),decision_tsv=str(run.output/name/'locus_decisions.tsv'),
                loci_file=str(run.output/name/'loci.txt'),scores_tsv=str(run.output/'scores.tsv'),
                caster_tree=str(run.output/name/'caster.nwk') if (directory/'caster.nwk').exists() else ''))
        if args.keep_diagnostics: write_json(final/'diagnostics/threshold_details.json',results)
        table(final/'results.tsv',result_rows)
        if bank: bank.path.unlink()
        progress.stage('publishing')
        receipt['status']='completed'
        receipt['wall_seconds_before_publication']=time.monotonic()-started
        receipt['resources']=monitor.finish()
        if args.build_filtered_tree or not run.guide_tree:
            receipt['binaries']['caster']=sha256(args.aster_dir/'bin/caster-site')
        publish(final,run.output,receipt)
        progress.finish('completed')
        completed=True
        return result_rows
    except BaseException as error:
        if progress is not None:
            progress.finish('failed',error=error)
        if monitor is not None and monitor.thread.is_alive(): receipt['resources']=monitor.finish()
        receipt.update(status='failed',error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        try:
            final.mkdir(exist_ok=True)
            write_json(final/'error.json',receipt)
        except OSError:
            pass  # Preserve the original error if disk exhaustion prevents diagnostics.
        if not run.output.exists():
            try: publish(final,run.output,receipt)
            except OSError: pass
        else:
            # Only a directory exclusively reserved by publish is ours. Preflight
            # conflicts are rejected before execute_run; never remove target data.
            failure=run.output.parent/(run.output.name+'.failure.json')
            if not failure.exists(): write_json(failure,receipt)
        raise
    finally:
        if progress is not None and not completed and progress.payload['status'] == 'running':
            progress.finish('failed',error='run terminated before completion')
        if completed or not args.keep_failed_temp: shutil.rmtree(scratch)


def execute(args,runs,specs):
    for run in runs:
        print(f'run {run.run_id}: loci={len(run.paths)} output={run.output}',flush=True)
    if args.dry_run:
        print(f'scratch={args.scratch_root}; threads={args.threads}; internal_budget_gib={args.memory_budget_gib}; batch_limit={args.msa_batch_size}')
        return []
    if args.build_statistics:
        subprocess.run(['make','-C',str(project_root()/'Afresh'),'iils-statistics',f'ASTER_DIR={args.aster_dir}'],check=True)
    if not args.statistics_bin.is_file(): raise FileNotFoundError(args.statistics_bin)
    if any(not r.guide_tree for r in runs) or args.build_filtered_tree:
        if not (args.aster_dir/'bin/caster-site').is_file(): raise FileNotFoundError(args.aster_dir/'bin/caster-site')
    batch=[]
    states={r.run_id:'not_executed' for r in runs}
    if args.run_manifest:
        args.out_root.mkdir(parents=True,exist_ok=False)  # Own the batch before starting any run.
    previous=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda _signum,_frame: (_ for _ in ()).throw(InterruptedError('received SIGTERM')))
    try:
        for run in runs:
            try:
                batch.extend(execute_run(args,run,specs)); states[run.run_id]='completed'
            except BaseException:
                states[run.run_id]='failed'; raise
    finally:
        signal.signal(signal.SIGTERM,previous)
        if args.run_manifest:
            args.out_root.mkdir(parents=True,exist_ok=True)
            path=args.out_root/'batch_receipt.json'
            if not path.exists(): write_json(path,dict(schema='iils-one-shot-batch-v1',runs=states))
    return batch
