"""Bounded synchronous IPC to a persistent Afresh statistics process."""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np

MAGIC, VERSION = 0x534c4949, 1
HEADER = struct.Struct('<IIIQQ')
MAX_FRAME = 1 << 30


class ProtocolError(RuntimeError):
    pass


class Buffer:
    def __init__(self, data=b''):
        self.data = bytearray(data)
        self.pos = 0

    def put(self, fmt, *values):
        self.data.extend(struct.pack('<' + fmt, *values))

    def string(self, value):
        raw = value.encode() if isinstance(value, str) else value
        self.put('Q', len(raw))
        self.data.extend(raw)

    def get(self, fmt):
        size = struct.calcsize('<' + fmt)
        if size > len(self.data) - self.pos:
            raise ProtocolError('truncated numeric payload')
        values = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += size
        return values[0] if len(values) == 1 else values

    def read_string(self):
        n = self.get('Q')
        if n > len(self.data) - self.pos:
            raise ProtocolError('truncated string payload')
        raw = self.data[self.pos:self.pos + n]
        self.pos += n
        return raw.decode()

    def array(self, dtype, count):
        size = np.dtype(dtype).itemsize * count
        if count < 0 or size > len(self.data) - self.pos:
            raise ProtocolError('truncated array payload')
        result = np.frombuffer(self.data, dtype=dtype, count=count, offset=self.pos).copy()
        self.pos += size
        return result

    def finish(self):
        if self.pos != len(self.data):
            raise ProtocolError('unexpected trailing payload')


class NativeWorker:
    def __init__(self, binary: Path, guide: Path, *, threads: int, seed: int, log: Path,
                 mapping_path: Path | None = None, max_frame: int = MAX_FRAME):
        self.max_frame = min(max_frame, MAX_FRAME)
        self.log = log.open('wb')
        try:
            command = [str(binary), '--serve', '-c', str(guide)]
            if mapping_path is not None:
                command.extend(['-a', str(mapping_path)])
            command.extend(['--threads', str(threads), '--seed', str(seed)])
            self.process = subprocess.Popen(command,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        except BaseException:
            self.log.close()
            raise
        self.request_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass  # preserve the original worker failure during cleanup
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
        self.log.close()

    def _read(self, n):
        result = bytearray()
        while len(result) < n:
            chunk = self.process.stdout.read(n - len(result))
            if not chunk:
                raise ProtocolError(f'worker terminated or truncated response {self.request_id}')
            result.extend(chunk)
        return result

    def request(self, operation: int, payload: Buffer) -> Buffer:
        if len(payload.data) > self.max_frame:
            raise ValueError('native request exceeds configured frame budget')
        self.request_id += 1
        try:
            self.process.stdin.write(HEADER.pack(MAGIC, VERSION, operation, self.request_id, len(payload.data)))
            self.process.stdin.write(payload.data)
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise ProtocolError('worker request pipe failed') from exc
        magic, version, status, request_id, n = HEADER.unpack(self._read(HEADER.size))
        if (magic, version, request_id) != (MAGIC, VERSION, self.request_id) or n > self.max_frame:
            raise ProtocolError('invalid worker response identity or size')
        response = Buffer(self._read(n))
        if status:
            message = response.read_string()
            response.finish()
            raise ProtocolError(f'worker request {request_id} failed: {message}')
        return response

    def msa(self, entries):
        """Entries contain original locus index, path and already-hashed FASTA bytes."""
        payload = Buffer()
        payload.put('I', len(entries))
        expected_ids = [index for index, _, _ in entries]
        for index, path, raw in entries:
            payload.put('I', index)
            payload.string(str(path))
            payload.string(raw)
        response = self.request(1, payload)
        n, s, t = response.get('III')
        if n != len(entries) or s > t or t > 100000:
            raise ProtocolError('invalid MSA completion counts')
        names = tuple(response.read_string() for _ in range(t))
        metadata = []
        for expected in expected_ids:
            index, raw, used, blocks, present = response.get('IQQII')
            if index != expected or present > t:
                raise ProtocolError('invalid MSA locus identity')
            taxa = response.array('<u4', present)
            if len(set(taxa)) != present or np.any(taxa >= t):
                raise ProtocolError('invalid present taxon IDs')
            metadata.append((index, raw, used, blocks, taxa))
        keys, blocks = [], []
        dtype = np.dtype([('scores','<f8',(4,)), ('capacity','<u8')])
        for _ in range(s):
            keys.append(response.read_string())
            blocks.append(response.array(dtype,n))
        response.finish()
        if len(set(keys)) != s:
            raise ProtocolError('duplicate split in MSA response')
        records = np.asarray(blocks, dtype=dtype)
        if not np.all(np.isfinite(records['scores'])):
            raise ProtocolError('nonfinite MSA statistics')
        return names, metadata, keys, records

    def open_bank(self, bank):
        payload = Buffer()
        payload.string(str(bank.path))
        payload.put('II',bank.count,len(bank.taxa))
        response = self.request(2,payload)
        if response.get('II') != (bank.count,len(bank.taxa)):
            raise ProtocolError('bank completion identity mismatch')
        response.finish()

    def losses(self, tasks, split_keys):
        """A task is (task_id, present_taxon_ids, [(observation, global_gene_ids)])."""
        payload = Buffer()
        payload.put('I',len(tasks))
        expected = []
        for task_id, present, targets in tasks:
            genes = sorted({int(g) for _, ids in targets for g in ids})
            payload.put('QI',task_id,len(present))
            for taxon in present:
                payload.put('I',int(taxon))
            payload.put('I',len(genes))
            for gene in genes:
                payload.put('I',gene)
            payload.put('I',len(targets))
            for locus, ids in targets:
                payload.put('III',locus.index,locus.denominator,len(locus.split_ids))
                for sid, weight in zip(locus.split_ids,locus.weights,strict=True):
                    payload.string(split_keys[sid])
                    payload.put('d',float(weight))
                payload.put('I',len(ids))
                for gene in ids:
                    payload.put('I',int(gene))
            expected.append((task_id,len(genes),targets))
        response = self.request(3,payload)
        if response.get('I') != len(expected):
            raise ProtocolError('incomplete loss task batch')
        completed = []
        dtype = np.dtype([('gene_id','<u4'),('loss','<f8')])
        for task_id, genes, targets in expected:
            if response.get('QII') != (task_id,genes,len(targets)):
                raise ProtocolError('loss completion identity mismatch')
            for locus, ids in targets:
                if response.get('II') != (locus.index,len(ids)):
                    raise ProtocolError('loss locus identity mismatch')
                values = response.array(dtype,len(ids))
                if not np.array_equal(values['gene_id'],ids) or not np.all(np.isfinite(values['loss'])):
                    raise ProtocolError('invalid global gene IDs or loss values')
                completed.append((locus.index,np.asarray(ids),values['loss'].copy()))
        response.finish()
        return completed
