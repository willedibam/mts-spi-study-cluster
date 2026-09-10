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


def test_partial_http_response_is_discarded_before_retry():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading
    payload=bytes(range(256))*16
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.headers.get('Range'))
            self.send_response(206)
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload[:2000] if len(calls)==1 else payload)
            self.close_connection=True
        def log_message(self,*args):pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        assert nz.get_range(f'http://127.0.0.1:{server.server_port}/signal',0,len(payload))==payload
        assert len(calls)==2
    finally:
        server.shutdown();server.server_close();thread.join()
