"""Audit public archive availability and bounded annotation-only ZIP reads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import zipfile
import zlib

import numpy as np
from scipy.io import loadmat


def request(url, *, byte_range=None, head=False):
    cmd = ["curl", "-fsSL", "--max-time", "25"]
    if head:
        cmd.append("--head")
    else:
        cmd += ["--max-filesize", "2097152"]
    if byte_range is not None:
        cmd += ["--range", byte_range]
    return subprocess.run(cmd + [url], check=True, capture_output=True).stdout


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if not k.startswith("__")}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def annotations(url, destination):
    tail = request(url, byte_range="-65536")
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise ValueError("no ZIP end record in bounded tail")
    end = struct.unpack("<4s4H2LH", tail[offset:offset + 22])
    _, disk, cd_disk, disk_entries, entries, cd_size, cd_offset, comment = end
    if disk or cd_disk or disk_entries != entries or cd_size > 2097152 or offset + 22 + comment != len(tail):
        raise ValueError("unsupported ZIP layout")
    cd = tail[offset-cd_size:offset] if cd_size <= offset else request(url, byte_range=f"{cd_offset}-{cd_offset+cd_size-1}")
    synthetic_end = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, entries, entries, cd_size, 0, 0)
    with zipfile.ZipFile(io.BytesIO(cd + synthetic_end)) as archive:
        members = archive.infolist()
    selected = [x for x in members if Path(x.filename).name.lower() in {"condition.mat", "event.mat"}
                and "__MACOSX" not in x.filename]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "members.json").write_text(json.dumps([dict(name=x.filename, size=x.file_size,
        compressed_size=x.compress_size) for x in members], indent=2) + "\n")
    result = []
    for index, member in enumerate(selected):
        if member.file_size > 2097152 or member.compress_size > 2097152:
            raise ValueError("annotation member exceeds read budget")
        header = request(url, byte_range=f"{member.header_offset}-{member.header_offset+29}")
        if header[:4] != b"PK\x03\x04" or len(header) != 30:
            raise ValueError("range request did not return a local ZIP header")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        start = member.header_offset + 30 + name_len + extra_len
        compressed = request(url, byte_range=f"{start}-{start+member.compress_size-1}")
        if len(compressed) != member.compress_size:
            raise ValueError("range length mismatch")
        if member.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(compressed, -15)
        elif member.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise ValueError("unsupported compression")
        if len(raw) != member.file_size or zlib.crc32(raw) != member.CRC:
            raise ValueError("member size/CRC mismatch")
        (destination / f"annotation-{index}.mat").write_bytes(raw)
        result.append(dict(member=member.filename, sha256=hashlib.sha256(raw).hexdigest(),
                           variables=plain(loadmat(io.BytesIO(raw), simplify_cells=True))))
    return dict(total_members=len(members), annotations=result)


def main(root):
    output = root / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    catalogue = json.loads((root / "detail.json").read_text())["expdata"]
    rows = [r for r in catalogue if r["Task"] == "Anesthesia and Sleep"]
    def inspect(row):
        item = dict(name=row["Name"], animal=row["Monkey"], agent=row["Session"], date=row["Date"])
        link = next(f["filename"] for f in row["Download"] if f["format"] == "mat")
        item["url"] = link
        try:
            headers = request(link, head=True).decode()
            item["http_status"] = int(re.findall(r"HTTP/\S+\s+(\d+)", headers)[-1])
            lengths = re.findall(r"(?im)^content-length:\s*(\d+)", headers)
            item["bytes"] = int(lengths[-1]) if lengths else None
            item["headers"] = headers
        except Exception as exc:
            item["error"] = str(exc)
        return item
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(inspect, rows))
    (root / "availability.json").write_text(json.dumps(checks, indent=2) + "\n")
    seen, sampled = set(), []
    for record in checks:
        key = record["animal"], record["agent"]
        if record["agent"] not in {"KTMD", "PF"} or key in seen:
            continue
        seen.add(key)
        sample = {k: record[k] for k in ["name", "animal", "agent", "url"]}
        try:
            sample.update(annotations(record["url"], root / record["name"]))
        except Exception as exc:
            sample["error"] = str(exc)
        sampled.append(sample)
        print(f"Inspected annotations: {key}", flush=True)
    report = dict(catalogue_sha256=hashlib.sha256((root / "detail.json").read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        selection="exact Task field; legacy task=40 is Sleep, task=78 is Anesthesia and Sleep",
        records=len(rows), availability=checks, annotation_samples=sampled,
        note="archive entries are not independent animals; no waveform data downloaded")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(dict(records=len(rows), accessible=sum(x.get("http_status") == 200 for x in checks),
               annotation_samples=len(sampled), failures=sum("error" in x for x in sampled)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
