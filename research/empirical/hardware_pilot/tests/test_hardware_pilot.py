import io
import zipfile
import zlib

import pytest
import numpy as np

from research.empirical.hardware_pilot import download_hardware_pilot as download
from research.empirical.hardware_pilot.hardware_pilot import audit_record, resample_record


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


def test_download_resumes_compressed_prefix(tmp_path, monkeypatch):
    data, raw = example_zip()
    entry = download.directory_entries(data[data.index(b'PK\x01\x02'):])[0]
    offset = 30 + len(entry['name'])
    cache = tmp_path / 'member.partial'
    cache.write_bytes(data[offset:offset + 5])
    starts = []
    def read(url, start, length):
        starts.append(start)
        return data[start:start + length]
    monkeypatch.setattr(download, 'request', read)
    reader = download.SplitZip([{'key':'x.zip', 'size':len(data), 'links':{'self':'0'}}])
    assert reader.extract(entry, cache) == raw
    assert offset + 5 in starts


def test_verified_range_with_short_body_is_retried_without_accepting_partial_data(monkeypatch):
    class Response:
        status = 206
        headers = {'Content-Range': 'bytes 10-13/20'}
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return self.data
    responses = iter([Response(b'ab'), Response(b'abcd')])
    monkeypatch.setattr(download.urllib.request, 'urlopen', lambda *a, **k: next(responses))
    monkeypatch.setattr(download.time, 'sleep', lambda seconds: None)
    assert download.request('https://example.test/public.dat', 10, 4) == b'abcd'


def test_truncated_range_retries_are_bounded(monkeypatch):
    class Response:
        status = 206
        headers = {'Content-Range': 'bytes 10-13/20'}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self, limit): return b'ab'
    calls = []
    def open_response(*args, **kwargs):
        calls.append(1)
        return Response()
    monkeypatch.setattr(download.urllib.request, 'urlopen', open_response)
    monkeypatch.setattr(download.time, 'sleep', lambda seconds: None)
    with pytest.raises(OSError, match='Truncated range body'):
        download.request('https://example.test/public.dat', 10, 4)
    assert len(calls) == 3


def test_sampling_preserves_duration_and_rejects_lost_carrier():
    fs = 2000
    time = np.arange(20000) / fs
    # Low-frequency case survives a full-duration conversion to 200 Hz.
    low = np.array([np.sin(2*np.pi*12*time + phase) + .08*np.sin(2*np.pi*25*time)
                    for phase in (.1, .7, 1.4, 2.1)])
    report, views, _ = audit_record(low, fs, targets=(2000,))
    assert views[2000].shape == (4, 2000)
    assert report[0]['seconds'] == 10
    assert report[0]['passes']
    # Anti-alias filtering correctly removes this carrier; it must NOT pass.
    high = np.array([np.sin(2*np.pi*160*time + phase) + .2*np.sin(2*np.pi*15*time)
                     for phase in (.1, .7, 1.4, 2.1)])
    report, _, _ = audit_record(high, fs, targets=(2000, 5000))
    assert not report[0]['passes']
    assert report[0]['reconstruction_nrmse_p95'] > .9
    assert report[1]['passes']


def test_nonintegral_resampling_is_explicit():
    with pytest.raises(ValueError, match='integer'):
        resample_record(np.ones((3, 21)), 10)
