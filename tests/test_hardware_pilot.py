import io
import zipfile
import zlib

import pytest

from scripts import download_hardware_pilot as download


def example_zip():
    buf = io.BytesIO()
    raw = b"1 2\n3 4\n" * 200
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("Random/ST_0.dat", raw)
    return buf.getvalue(), raw


def test_range_extract_across_split_boundaries(monkeypatch):
    data, raw = example_zip()
    entry = download.directory_entries(data[data.index(b"PK\x01\x02"):])[0]
    blocks = [data[i:i + 37] for i in range(0, len(data), 37)]
    files = [{"key": f"x.z{i + 1:02d}" if i < len(blocks)-1 else "x.zip",
              "size": len(b), "links": {"self": str(i)}} for i, b in enumerate(blocks)]
    monkeypatch.setattr(download, "request", lambda url, start, length: blocks[int(url)][start:start + length])
    reader = download.SplitZip(files)
    assert reader.extract(entry) == raw
    with pytest.raises(ValueError, match="CRC"):
        reader.extract(dict(entry, crc32=zlib.crc32(raw) ^ 1))


def test_single_disk_inventory(monkeypatch):
    data, raw = example_zip()
    monkeypatch.setattr(download, "request", lambda url, start, length: data[start:start + length])
    reader = download.SplitZip([{"key": "x.zip", "size": len(data), "links": {"self": "0"}}])
    entries = reader.inventory()
    assert len(entries) == 1
    assert reader.extract(entries[0]) == raw
