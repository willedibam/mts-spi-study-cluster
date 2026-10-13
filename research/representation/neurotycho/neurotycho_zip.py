"""Bounded, verified reads of individual members in public NeuroTycho ZIPs."""
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import time
import zipfile
import zlib


def get_range(url, start, length):
    if length < 1 or length > 64 * 1024**2:
        raise ValueError("range outside 64 MiB member budget")
    span = f"-{length}" if start is None else f"{start}-{start + length - 1}"
    command = ["curl", "-fsSL", "--connect-timeout", "20",
         "--max-time", "180", "--max-filesize", str(max(length, 2 * 1024**2)),
         "--range", span, url]
    # curl's internal retries can append partial bodies to captured stdout.
    # A fresh process per attempt discards partial bytes before retrying.
    for attempt in range(3):
        try:
            result = subprocess.run(command, check=True, capture_output=True).stdout
            if len(result) != length:
                raise ValueError(f"range length mismatch: {len(result)} != {length}")
            return result
        except (subprocess.CalledProcessError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def directory(url):
    tail = get_range(url, None, 65536)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise ValueError("missing ZIP end record")
    _, disk, cd_disk, n_disk, n, size, start, comment = struct.unpack(
        "<4s4H2LH", tail[offset:offset + 22])
    if disk or cd_disk or n != n_disk or size > 2 * 1024**2 or offset + 22 + comment != len(tail):
        raise ValueError("unsupported ZIP layout")
    cd = tail[offset-size:offset] if size <= offset else get_range(url, start, size)
    end = struct.pack("<4s4H2LH", b"PK\x05\x06", 0, 0, n, n, size, 0, 0)
    with zipfile.ZipFile(io.BytesIO(cd + end)) as archive:
        return archive.infolist()


def read_member(url, member, destination):
    """Cache only complete, CRC-verified bytes; never extract archive paths."""
    destination = Path(destination)
    if member.file_size > 64 * 1024**2 or member.flag_bits & 1:
        raise ValueError("oversized or encrypted ZIP member")
    if destination.exists():
        raw = destination.read_bytes()
    else:
        header = get_range(url, member.header_offset, 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("missing local ZIP header")
        n_name, n_extra = struct.unpack("<HH", header[26:30])
        compressed = get_range(url, member.header_offset + 30 + n_name + n_extra,
                               member.compress_size)
        if member.compress_type == zipfile.ZIP_DEFLATED:
            decoder = zlib.decompressobj(-15)
            raw = decoder.decompress(compressed, member.file_size + 1)
            if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
                raise ValueError("invalid deflate stream")
        elif member.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise ValueError("unsupported ZIP compression")
    if len(raw) != member.file_size or zlib.crc32(raw) != member.CRC:
        raise ValueError(f"member size/CRC mismatch: {member.filename}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(raw)
        temporary.replace(destination)
    provenance = dict(url=url, member=member.filename, bytes=len(raw), crc32=member.CRC,
                      sha256=hashlib.sha256(raw).hexdigest())
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(provenance, indent=2) + "\n")
    return raw
