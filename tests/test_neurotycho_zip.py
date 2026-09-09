import io
import struct
import zipfile

import pytest

from src import neurotycho_zip as nz


def test_ranges_roundtrip_and_cache_corruption(tmp_path, monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('session/signal.mat', bytes(range(256)) * 400)
        archive.writestr('padding', b'x' * 70000, compress_type=zipfile.ZIP_STORED)
    raw_zip = stream.getvalue()
    calls = []
    def get_range(url, start, length):
        calls.append((start, length))
        return raw_zip[-length:] if start is None else raw_zip[start:start+length]
    monkeypatch.setattr(nz, 'get_range', get_range)
    member = next(m for m in nz.directory('test') if m.filename.endswith('.mat'))
    path = tmp_path / 'cached.mat'
    expected = bytes(range(256)) * 400
    assert nz.read_member('test', member, path) == expected
    count = len(calls)
    assert nz.read_member('test', member, path) == expected
    assert len(calls) == count
    path.write_bytes(expected[:-1] + b'!')
    with pytest.raises(ValueError, match='CRC'):
        nz.read_member('test', member, path)


def test_oversize_rejected_before_fetch(tmp_path, monkeypatch):
    member = zipfile.ZipInfo('large')
    member.file_size = 65 * 1024**2
    monkeypatch.setattr(nz, 'get_range', lambda *a: pytest.fail('should not fetch'))
    with pytest.raises(ValueError, match='oversized'):
        nz.read_member('test', member, tmp_path / 'large')
