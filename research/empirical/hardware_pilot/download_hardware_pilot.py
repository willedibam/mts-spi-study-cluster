"""Download selected members of a public Zenodo split ZIP, checking ranges and CRC."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import urllib.request
import zlib


def request(url, start=None, length=None):
    headers = {"User-Agent": "mts-spi-study hardware feasibility", "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{start + length - 1}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as r:
                if start is not None:
                    expected = f"bytes {start}-{start + length - 1}/"
                    if r.status != 206 or not r.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError(f"Server did not honour byte range: {r.status}, {r.headers.get('Content-Range')}")
                limit = length if length is not None else 2_000_000
                data = r.read(limit + 1)
                if start is not None and len(data) < length:
                    # A valid 206 header can still precede a truncated body.
                    # Retry this range without admitting it to the saved prefix.
                    raise OSError(f"Truncated range body: expected {length} bytes, received {len(data)}")
                if len(data) > limit:
                    raise ValueError(f"Response exceeds requested length: limit {limit} bytes, received at least {len(data)}")
                return data
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def directory_entries(data):
    """Parse central entries, including ZIP64 offsets in split archives."""
    entries, pos = [], 0
    while pos + 46 <= len(data) and data[pos:pos + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<4s6H3I5H2I", data, pos)
        _, _, _, flags, method, _, _, crc, csize, size, nlen, xlen, clen, disk, _, _, offset = fields
        name = data[pos + 46:pos + 46 + nlen].decode("utf-8" if flags & 2048 else "cp437")
        extra = data[pos + 46 + nlen:pos + 46 + nlen + xlen]
        j = 0
        while j + 4 <= len(extra):
            kind, length = struct.unpack_from("<HH", extra, j)
            block = extra[j + 4:j + 4 + length]
            if kind == 1:
                k = 0
                values = [size, csize, offset, disk]
                for q, (sentinel, fmt) in enumerate([(0xffffffff, "Q")] * 3 + [(0xffff, "I")]):
                    if values[q] == sentinel:
                        values[q] = struct.unpack_from("<" + fmt, block, k)[0]
                        k += struct.calcsize(fmt)
                size, csize, offset, disk = values
            j += 4 + length
        entries.append(dict(name=name, flags=flags, method=method, crc32=crc,
                            compressed_size=csize, size=size, disk=disk, offset=offset))
        pos += 46 + nlen + xlen + clen
    if not entries:
        raise ValueError("No central-directory entries")
    return entries


class SplitZip:
    def __init__(self, files, chunk_bytes=8 * 1024 * 1024, workers=4):
        parts = sorted([f for f in files if re.search(r"\.z\d+$", f["key"], re.I)], key=lambda f: int(f["key"].rsplit("z", 1)[-1]))
        final = [f for f in files if f["key"].lower().endswith(".zip")]
        if len(final) != 1:
            raise ValueError("Expected one split ZIP set")
        self.parts = parts + final
        self.chunk_bytes = chunk_bytes
        self.workers = workers

    def read(self, disk, offset, length):
        chunks = []
        while length:
            part = self.parts[disk]
            if offset >= part["size"]:
                offset -= part["size"]
                disk += 1
                continue
            n = min(length, part["size"] - offset, self.chunk_bytes)
            url = part["links"].get("self") or part["links"]["content"]
            # Zenodo exposes the same public file through two routes. The API
            # content route returned 504 during this pilot; prefer the record URL.
            url = re.sub(r"/api/records/(\d+)/files/(.+)/content$", r"/records/\1/files/\2?download=1", url)
            chunks.append(request(url, offset, n))
            offset += n
            length -= n
        return b"".join(chunks)

    def inventory(self):
        disk = len(self.parts) - 1
        size = self.parts[-1]["size"]
        tail = self.read(disk, max(0, size - 65557), min(size, 65557))
        pos = tail.rfind(b"PK\x05\x06")
        if pos < 0:
            raise ValueError("ZIP end directory absent")
        _, last, cd_disk, _, count, length, offset, _ = struct.unpack_from("<4s4H2IH", tail, pos)
        if last != disk or max(length, offset) == 0xffffffff:
            raise ValueError("Unexpected ZIP disk layout or ZIP64 end directory")
        entries = directory_entries(self.read(cd_disk, offset, length))
        if len(entries) != count:
            raise ValueError("ZIP member count mismatch")
        return entries

    def extract(self, entry, cache=None):
        header = self.read(entry["disk"], entry["offset"], 30)
        if header[:4] != b"PK\x03\x04" or entry["flags"] & 1:
            raise ValueError("Invalid or encrypted member")
        nlen, xlen = struct.unpack_from("<HH", header, 26)
        data_offset = entry["offset"] + 30 + nlen + xlen
        if cache is None:
            compressed = self.read(entry["disk"], data_offset, entry["compressed_size"])
        else:
            # A transient HTTP failure should not discard an already transferred
            # prefix. The complete compressed member is CRC-checked below.
            size = cache.stat().st_size if cache.exists() else 0
            if size > entry['compressed_size']:
                raise ValueError('Compressed checkpoint exceeds member size')
            with cache.open('ab') as handle, ThreadPoolExecutor(max_workers=self.workers) as pool:
                while size < entry['compressed_size']:
                    offsets = range(size, min(size + self.workers * self.chunk_bytes, entry['compressed_size']), self.chunk_bytes)
                    def fetch(start):
                        n = min(self.chunk_bytes, entry['compressed_size'] - start)
                        return self.read(entry['disk'], data_offset + start, n)
                    # Ordered writes retain one contiguous resumable prefix.
                    # Bound concurrency and avoid hundreds of tiny HTTP requests.
                    for chunk in pool.map(fetch, offsets):
                        handle.write(chunk)
                        handle.flush()
                        size += len(chunk)
                    print(f"{entry['name']}: {size}/{entry['compressed_size']} compressed bytes", flush=True)
            compressed = cache.read_bytes()
        if entry["method"] == 8:
            raw = zlib.decompress(compressed, -15)
        elif entry["method"] == 0:
            raw = compressed
        else:
            raise ValueError(f"Unsupported compression: {entry['method']}")
        if len(raw) != entry["size"] or zlib.crc32(raw) != entry["crc32"]:
            raise ValueError("ZIP size/CRC verification failed")
        return raw


def main(record, root, steps):
    import numpy as np
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "zenodo-record.json"
    if not metadata_path.exists():
        data = request(f"https://zenodo.org/api/records/{record}")
        json.loads(data)
        metadata_path.write_bytes(data)
    metadata = json.loads(metadata_path.read_text())
    archive = SplitZip(metadata["files"])
    inventory_path = root / "zip-inventory.json"
    if not inventory_path.exists():
        inventory_path.write_text(json.dumps(archive.inventory(), indent=2) + "\n")
    entries = json.loads(inventory_path.read_text())
    selected = {}
    for entry in entries:
        match = re.search(r"(?:ST|TS)_(\d+)\.dat$", entry["name"], re.I)
        if match:
            number = int(match[1])
            if number in selected:
                raise ValueError(f"Duplicate setting {number}")
            selected[number] = entry
    print(json.dumps({"members": len(entries), "matched_settings": len(selected), "first_names": [e['name'] for e in entries[:5]]}), flush=True)
    for step in steps:
        entry = selected[step]
        target = root / f"step-{step:03d}.npz"
        if target.exists():
            continue
        cache = root / f'compressed-step-{step:03d}.partial'
        raw = archive.extract(entry, cache)
        # The published files contain exactly 28 numeric columns.
        x = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
        if x.shape != (500000, 28) or not np.isfinite(x).all():
            raise ValueError(f"Unexpected numeric contents: {entry['name']}: {x.shape}")
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(temporary, x=x.T, fs=50000, step=step, record=record)
        temporary.replace(target)
        receipt = dict(entry, record=record, raw_sha256=hashlib.sha256(raw).hexdigest(),
                       numeric_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        target.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
        cache.unlink()  # Own transient download, superseded by verified numeric data.
        print(f"Verified {record} step={step} shape={x.shape}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", type=int, default=17451096)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--steps", type=int, nargs="*", default=[])
    a = p.parse_args()
    main(a.record, a.root, a.steps)
