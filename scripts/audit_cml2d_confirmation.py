"""Independent local reconstruction and paired-input audit of completed outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cml2d_confirmation import FROZEN_HASHES, OLD_CONTROLS, confirmation_gate
from src.spi_spi_contract import build_unified_feature_values
from src.utils import slugify


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii")); h.update(str(a.shape).encode("ascii")); h.update(a.view(np.uint8))
    return h.hexdigest()


def audit(root, pilot):
    frozen = pilot / "primary-analysis"
    for name, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((frozen/name).read_bytes()).hexdigest() == expected
    with np.load(frozen/"model.npz", allow_pickle=False) as a:
        model = {name:a[name] for name in a.files}
    baseline = json.loads((frozen/"eligibility.json").read_text())["sources"][0]["execution_identity"]
    baseline = {k:v for k,v in baseline.items() if k != "corpus_config_sha256"}
    result = dict(frozen_model_arrays_identical=True, execution_core_matches_pilot=True,
        input_members_verified=0, paired_prefixes_verified=0, maximum_q_error=0.,
        maximum_missingness_error=0., replayed_MPI_rows=0, role="all confirmation rows are evaluation")
    manifests = {}
    for arm, count in (("primary",544),("sensitivity",864)):
        corpus, analysis = root/arm, root/f"{arm}-analysis"
        manifest = json.loads((corpus/"manifest.json").read_text())
        manifests[arm] = manifest
        assert len(manifest["rows"]) == count
        assert hashlib.sha256((corpus/"observations.npz").read_bytes()).hexdigest() == manifest["archive_sha256"]
        with np.load(analysis/"model.npz",allow_pickle=False) as a:
            for name, expected in model.items():
                np.testing.assert_array_equal(a[name], expected)
        with np.load(analysis/"features.npz",allow_pickle=False) as a:
            z = a["z"][:,model["keep"]]
            ids = a["row_id"].tolist()
            np.testing.assert_array_equal(a["spi_order"],model["spi_order"])
        scores = pd.read_csv(analysis/"scores.csv").set_index("row_id").loc[ids]
        assert set(scores.role) == {"evaluation"}
        assert set(scores.seed) == set(range(260911101,260911133))
        report = json.loads((analysis/"confirmation-report.json").read_text())
        assert confirmation_gate(scores) == report["gate"]
        missing = np.mean(~np.isfinite(z),axis=1)
        eligible = missing <= .05
        np.testing.assert_array_equal(eligible,scores.eligible)
        q = ((np.where(np.isfinite(z),z,model["impute"])-model["center"])@model["component"])/float(model["score_scale"])
        error = float(np.max(abs(q[eligible]-scores.q.to_numpy()[eligible])))
        missing_error = float(np.max(abs(missing-scores.selected_missingness)))
        assert error < 1e-10 and missing_error < 1e-14
        result["maximum_q_error"] = max(result["maximum_q_error"],error)
        result["maximum_missingness_error"] = max(result["maximum_missingness_error"],missing_error)
        identity = json.loads((analysis/"eligibility.json").read_text())
        assert identity["minimum_retained_per_cell"] == 24
        for source in identity["sources"]:
            core = {k:v for k,v in source["execution_identity"].items() if k != "corpus_config_sha256"}
            assert core == baseline
        # Independently rebuild complete z vectors from predetermined MPI examples.
        selected_indices = [1,225,544] if arm == "primary" else [1,2,3,432,864]
        for index in selected_indices:
            row = manifest["rows"][index-1]
            path = corpus/"mpi"/arm/f"{index:04d}-{slugify(row['row_id'],'dataset')}"/"spi_mpis.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sources"][index-1]["mpi_sha256"]
            with np.load(path,allow_pickle=False) as matrices:
                rebuilt,_,_ = build_unified_feature_values(matrices,model["spi_order"].tolist())
            np.testing.assert_allclose(rebuilt[model["keep"]],z[ids.index(row["row_id"])],rtol=0,atol=1e-12,equal_nan=True)
            result["replayed_MPI_rows"] += 1
        meta_paths = list((corpus/"mpi"/arm).glob("*/meta.json"))
        assert len(meta_paths) == count
        metas = {m["dataset_name"]:m for m in (json.loads(p.read_text()) for p in meta_paths)}
        with np.load(corpus/"observations.npz",allow_pickle=False) as raw:
            for row in manifest["rows"]:
                meta = metas[row["row_id"]]
                assert meta["status"] == "complete" and len(meta["pyspi"]["spis"]) == 289
                assert meta["source"]["archive_sha256"] == manifest["archive_sha256"]
                assert array_hash(raw[row["row_id"]]) == meta["source"]["member_sha256"]
                assert meta["M"] == row["M"] and meta["T"] == row["T"]
                result["input_members_verified"] += 1
        result[f"{arm}_gate_passes"] = report["gate"]["passes"]
    primary = {(r["seed"],r["r"]):r for r in manifests["primary"]["rows"]}
    with np.load(root/"primary/observations.npz",allow_pickle=False) as large, np.load(root/"sensitivity/observations.npz",allow_pickle=False) as small:
        for row in manifests["sensitivity"]["rows"]:
            parent = primary[row["seed"],row["r"]]
            assert row["r"] in OLD_CONTROLS
            assert row["master_sha256"] == parent["master_sha256"] and row["Q_reference"] == parent["Q_reference"]
            np.testing.assert_array_equal(small[row["row_id"]],large[parent["row_id"]][:row["M"],:row["T"]])
            result["paired_prefixes_verified"] += 1
    (root/"integrity-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--pilot",type=Path,required=True)
    a=p.parse_args();audit(a.root,a.pilot)
