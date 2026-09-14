"""Diagnose HCP p90 per-SPI time/memory with checkpoints; no catalogue changes."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import numpy as np
from pyspi.calculator import Calculator
from pyspi import _parallel
from src.run_external_corpus import ExternalCorpusConfig, load_inventory, load_timeseries


class TracedCalculator(Calculator):
    def _compute_serial(self, spi_keys, M, cp_dir, progress):
        # Same serial primitive/order/recording as pinned pyspi 65317c9.
        # Only add flushed observations and enable the library's atomic checkpoints.
        with self.trace_path.open('x') as stream:
            for index,key in enumerate(spi_keys):
                start=time.time()
                stream.write(json.dumps(dict(event='start',index=index,key=key,unix=start,
                                             peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n');stream.flush()
                S,err,warns,elapsed=_parallel.run_spi(self._spis[key],self.dataset,key,M)
                self._record(key,S,err,warns,elapsed)
                _parallel.write_checkpoint(cp_dir,key,S,err)
                stream.write(json.dumps(dict(event='complete',index=index,key=key,unix=time.time(),seconds=elapsed,
                                             error=err,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n');stream.flush()


def main(config_path,index,output):
    config=ExternalCorpusConfig.from_file(config_path)
    entry=load_inventory(config)[index-1]
    assert entry.index==index
    data,source=load_timeseries(config,entry)
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'trace.jsonl').exists(), 'Preserve diagnostic history'
    np.random.seed(config.random_seed);random.seed(config.random_seed)
    calc=TracedCalculator(dataset=data.T,config=str(config.pyspi_config),zscore=config.normalise,verbose=False)
    calc.trace_path=output/'trace.jsonl'
    (output/'identity.json').write_text(json.dumps(dict(name=entry.name,M=entry.M,T=entry.T,source=source,
        run_digest=calc.run_digest,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fresh diagnostic after memory failure; identical data/catalogue/SPI order/seed, instrumented serial loop, no completed production outputs overwritten.'),indent=2)+'\n')
    calc.compute(n_jobs=1,checkpoint_dir=output/'checkpoints',resume=False,progress=False)
    (output/'summary.json').write_text(json.dumps(dict(status='complete',timings=calc.timings,errors=calc.errors,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.config,args.index,args.output)
