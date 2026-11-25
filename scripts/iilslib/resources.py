"""Sample the Linux process tree and private workspace without extra packages."""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time


class ResourceMonitor:
    def __init__(self, root: Path, interval: float = .2):
        self.root=root
        self.interval=interval
        self.started=time.monotonic()
        self.stop=threading.Event()
        self.peak_rss=self.peak_files=self.peak_bytes=0
        self.cpu={}
        self.io={}
        self.thread=threading.Thread(target=self._loop,daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(self.interval)

    def sample(self):
        pending=[os.getpid()]
        visited=set()
        rss=0
        while pending:
            pid=pending.pop()
            if pid in visited: continue
            visited.add(pid)
            base=Path('/proc')/str(pid)
            try:
                stat=(base/'stat').read_text().rsplit(')',1)[1].split()
                key=(pid,stat[19])
                rss+=int(stat[21])*os.sysconf('SC_PAGE_SIZE')
                self.cpu[key]=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK')
            except (OSError,ValueError,IndexError):
                continue
            try:
                values=dict(line.split(':',1) for line in (base/'io').read_text().splitlines())
                self.io[key]={k:int(values[k]) for k in ('read_bytes','write_bytes','rchar','wchar')}
            except (OSError,ValueError,KeyError):
                pass
            try:
                for task in (base/'task').iterdir():
                    pending.extend(int(x) for x in (task/'children').read_text().split())
            except (OSError,ValueError,IndexError):
                pass
        self.peak_rss=max(self.peak_rss,rss)
        count=size=0
        for path in self.root.rglob('*'):
            try:
                if path.is_file(): count+=1; size+=path.stat().st_size
            except OSError: pass
        self.peak_files=max(self.peak_files,count)
        self.peak_bytes=max(self.peak_bytes,size)

    def finish(self):
        self.stop.set()
        self.thread.join()
        self.sample()
        return dict(wall_seconds=time.monotonic()-self.started,
            sampled_process_tree_peak_rss_bytes=self.peak_rss,
            sampled_private_scratch_peak_files=self.peak_files,
            sampled_private_scratch_peak_bytes=self.peak_bytes,
            sample_interval_seconds=self.interval,observed_processes=len(self.cpu),
            cpu_seconds_last_samples=list(self.cpu.values()),io_last_samples=list(self.io.values()),
            limitations='Sampled lower bounds; short-lived children may be missed. Linux parent IO can include waited children, so IO samples must not be summed. Disk versus remote bytes are not distinguished by /proc. External CASTER RSS is measured but not capped.')
