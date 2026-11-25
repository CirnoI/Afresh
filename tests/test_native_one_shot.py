"""Native integration: batching, sparse global IDs, missing taxa, IPC failures."""
import csv
from io import StringIO
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import dendropy
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from iilslib.native_worker import NativeWorker,Buffer,ProtocolError
from iilslib.observations import observe, inventory
from iilslib.tree_bank import write_bank
from iilslib.null_scheduler import null_losses


@unittest.skipUnless((ROOT/'Afresh/bin/iils-statistics').is_file(),'build Afresh first')
class NativeIntegration(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.binary=ROOT/'Afresh/bin/iils-statistics'
        self.guide=self.root/'guide.nwk'
        self.guide.write_text("(A,(B,(C,(D,(E,F)'[s1=70;s2=20;s3=10]':0.5)'[s1=60;s2=30;s3=10]':0.3)'[s1=50;s2=25;s3=25]':0.4));\n")
        rng=random.Random(11)
        self.paths=[]
        for i in range(5):
            path=self.root/f'{i}.fa'
            path.write_text(''.join(f'>{t}\n'+''.join(rng.choices('ACGT',k=120))+'\n'
                                    for t in ('ABCDEF' if i<3 else 'ABCDE')))
            self.paths.append(path)

    def worker(self,seed=0,mapping=None):
        return NativeWorker(self.binary,self.guide,threads=1,seed=seed,
                            mapping_path=mapping,log=self.root/'worker.log')

    def test_msa_batch_maps_alignment_labels_to_guide_taxa(self):
        mapped = self.root / 'name-map.tsv'
        mapped.write_text(''.join(f'{taxon}_copy {taxon}\n' for taxon in 'ABCDEF'))
        paths = []
        for index in range(2):
            path = self.root / f'mapped-{index}.fa'
            path.write_text(''.join(
                f'>{taxon}_copy\n' + ''.join(random.Random(index + ord(taxon)).choices('ACGT', k=120)) + '\n'
                for taxon in 'ABCDEF'
            ))
            paths.append(path)
        with self.worker(mapping=mapped) as worker:
            observations = observe(paths, self.guide, worker, transform='raw-alt',
                                   batch_size=2, memory_bytes=64 << 20)
        self.assertEqual(observations.taxa, tuple('ABCDEF'))
        self.assertTrue(all(locus.scores['n_taxa_present'] == 6 for locus in observations.loci))

    def test_msa_batch_reordering_and_budget(self):
        with self.worker() as worker:
            a=observe(self.paths,self.guide,worker,transform='raw-alt',batch_size=1,memory_bytes=64<<20)
            b=observe(self.paths,self.guide,worker,transform='raw-alt',batch_size=10,memory_bytes=64<<20)
            for x,y in zip(a.loci,b.loci):
                self.assertEqual(x.scores,y.scores)
                np.testing.assert_array_equal(x.weights,y.weights)
            names,metadata,keys,records=worker.msa([(17,self.paths[1],self.paths[1].read_bytes()),
                                                   (3,self.paths[0],self.paths[0].read_bytes())])
            self.assertEqual([m[0] for m in metadata],[17,3])
            with self.assertRaisesRegex(ValueError,'single alignment'):
                observe(self.paths,self.guide,worker,transform='raw-alt',batch_size=1,memory_bytes=32)

    def test_polytomy_global_ids_and_dense_scheduler(self):
        # Compare each sparse ID against the standalone exporter using its true offset.
        tree='(A,B,C,(D,E,F));'
        trees=dendropy.TreeList.get(data='\n'.join([tree]*40),schema='newick',preserve_underscores=True)
        with self.worker(seed=7) as worker:
            obs=observe(self.paths,self.guide,worker,transform='raw-alt',batch_size=3,memory_bytes=64<<20)
            bank=write_bank(self.root/'bank',iter(trees),40,obs.taxa)
            worker.open_bank(bank)
            targets=[(l,np.array([3,17,39],dtype=np.uint32)) for l in obs.loci[:3]]
            sparse=worker.losses([(0,list(range(6)),targets)],obs.split_keys)
            for global_id in (3,17,39):
                source=self.root/'one.tre'; source.write_text(tree+'\n')
                result=subprocess.run([str(self.binary),'--input-type','gene-tree','-c',str(self.guide),
                    '--seed','7','--gene-id-offset',str(global_id),str(source)],capture_output=True,text=True,check=True)
                masses={r['split_key']:float(r['discordance_mass']) for r in csv.DictReader(StringIO(result.stdout),delimiter='\t')}
                for locus,(_,ids,values) in zip(obs.loci,sparse):
                    expected=sum(w*masses.get(obs.split_keys[s],0) for s,w in zip(locus.split_ids,locus.weights))/locus.denominator
                    np.testing.assert_allclose(values[list(ids).index(global_id)],expected,rtol=1e-10,atol=1e-12)
        outputs={}
        stats=null_losses(obs,{l.index:np.arange(40,dtype=np.uint32) for l in obs.loci},bank,
                          binary=self.binary,guide=self.guide,seed=7,threads=2,memory_bytes=256<<20,work=self.root,
                          consume=lambda i,ids,array:outputs.update({i:array.copy()}))
        self.assertEqual(len(outputs),5)
        self.assertEqual(stats['gene_statistics'],80)
        for i,ids,values in sparse:
            np.testing.assert_allclose(outputs[i][ids],values,rtol=1e-10,atol=1e-12)

    def test_worker_error_is_not_zero(self):
        with self.worker() as worker:
            with self.assertRaisesRegex(ProtocolError,'unsupported request'):
                worker.request(999,Buffer())
        with self.worker() as worker:
            worker.process.kill(); worker.process.wait()
            with self.assertRaises(ProtocolError):
                worker.msa([(0,self.paths[0],self.paths[0].read_bytes())])


if __name__=='__main__': unittest.main()
