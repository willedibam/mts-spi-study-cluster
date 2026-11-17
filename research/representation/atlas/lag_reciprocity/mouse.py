"""Check the 20-ms macaque timing readout on the Claar wake/isoflurane windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import state_scores


def main(directory, output):
    records = []
    # Use the first date of each mouse, as in the original MI study.
    seen = set()
    for path in sorted(directory.glob("sub-*.npz")):
        animal = path.stem.split("_")[0]
        if animal in seen:
            continue
        with np.load(path) as bank:
            metadata = pd.DataFrame(json.loads(str(bank["rows"])))
            windows = bank["W"].astype(float)
        if min((metadata.state == label).sum() for label in ("awake", "iso")) < 10:
            continue
        seen.add(animal)
        for montage in ("CAR", "as recorded"):
            x = windows
            if montage == "CAR":
                x = (windows - windows.mean(axis=2, keepdims=True))[:, :, :-1]
            for state in ("awake", "iso", "early awake", "late awake"):
                indices = np.flatnonzero(metadata.state == ("iso" if state == "iso" else "awake"))
                indices = indices[np.argsort(metadata.iloc[indices].start_s)]
                n = len(indices) // 2
                if state == "early awake":
                    indices = indices[:n]
                if state == "late awake":
                    indices = indices[-n:]
                result = state_scores(x[indices], 2)
                records.append(dict(animal=animal, date=path.stem, montage=montage, state=state,
                                    n=len(indices), channels=x.shape[2], **result))
    pd.DataFrame(records).to_csv(output, index=False, float_format="%.10g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.directory, args.output)
