"""Release nine homogeneous farms from completed one-recording measurements."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

from mtsspi.provenance import atomic_json
from mtsspi.run_corpus import load_corpus, completion_error
from mtsspi.utils import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_jobs", type=Path)
    args = parser.parse_args()
    root = project_root()
    config = "research/proof/p90/configs/proof-p90-261014.yaml"
    corpus = load_corpus(root / config)
    entries = corpus.entries()
    output = corpus.base_output_dir.parent / "dispatch"
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pilot in json.loads(args.pilot_jobs.read_text()):
        entry = entries[pilot["index"] - 1]
        assert completion_error(corpus, entry) is None
        receipt_path = output / f"M{entry.M}_T{entry.T}.json"
        if receipt_path.exists():
            jobs.append(json.loads(receipt_path.read_text())["job"])
            continue
        status = subprocess.check_output(["qstat", "-xf", pilot["job"]], text=True)
        assert re.search(r"Exit_status\s*=\s*0\s", status), status
        memory_kb = int(re.search(r"resources_used.mem\s*=\s*(\d+)kb", status)[1])
        h, m, s = map(int, re.search(r"resources_used.walltime\s*=\s*([\d:]+)", status)[1].split(":"))
        measured_seconds = h*3600 + m*60 + s
        per_recording_gb = memory_kb / 1024**2 * 1.25
        selected = [e.index for e in entries if (e.M, e.T) == (entry.M, entry.T)]
        assert len(selected) == 120
        index_path = output / f"M{entry.M}_T{entry.T}.indices"
        index_path.write_text("\n".join(map(str, selected)) + "\n")
        cpr = pilot["cores_per_recording"]
        ncpus = math.ceil(len(selected)*cpr/48)*48
        nodes = ncpus//48
        workers = min(len(selected), nodes*min(48//cpr, math.floor(190/per_recording_gb)))
        if workers < 1:
            raise ValueError(f"Measured recording does not fit normal nodes: {pilot}")
        wall_seconds = max(600, math.ceil(2*measured_seconds*math.ceil(len(selected)/workers)/60)*60)
        walltime = f"{wall_seconds//3600:02d}:{wall_seconds%3600//60:02d}:00"
        environment = {**os.environ, "NCPUS": str(ncpus), "WORKERS": str(workers),
                       "CORES_PER_RECORDING": str(cpr), "MEM_GB": str(math.ceil(workers*per_recording_gb)),
                       "WALLTIME": walltime, "TASK_TIMEOUT": str(wall_seconds-60)}
        response = subprocess.check_output(["bash", "jobs/gadi/submit_farm.sh", config, str(index_path)],
                                           env=environment, text=True, cwd=root)
        job = re.search(r"\b\d+\.gadi-pbs\b", response)[0]
        receipt = {**pilot, "job": job, "pilot_job": pilot["job"], "M": entry.M, "T": entry.T,
                   "ncpus": ncpus, "workers": workers, "mem_gb": int(environment["MEM_GB"]),
                   "walltime": walltime, "measured_seconds": measured_seconds,
                   "measured_memory_kb": memory_kb}
        atomic_json(receipt_path, receipt)
        jobs.append(job)
        print(json.dumps(receipt), flush=True)
    figure_receipt = output / "figures.json"
    if not figure_receipt.exists():
        job = subprocess.check_output([
            "qsub", "-W", "depend=afterok:"+":".join(jobs),
            "-v", "PYSPI_SOURCE="+os.environ["PYSPI_SOURCE"],
            "research/proof/p90/jobs/finish_current_refresh.pbs",
        ], text=True, cwd=root).strip()
        atomic_json(figure_receipt, {"job": job, "afterok": jobs})
        print(f"Figures: {job}", flush=True)


if __name__ == "__main__":
    main()
