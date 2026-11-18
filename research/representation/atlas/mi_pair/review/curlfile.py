"""Read-only file-like object over HTTP range requests via curl (system TLS), with 1-MB block cache."""
import subprocess, io
class CurlFile(io.RawIOBase):
    B = 2 ** 20
    def __init__(self, url):
        self.url = subprocess.run(['curl', '-sIL', '-o', '/dev/null', '-w', '%{url_effective}', url], capture_output=True, text=True).stdout
        h = subprocess.run(['curl', '-sI', self.url], capture_output=True, text=True).stdout
        self.size = int([l.split(':')[1] for l in h.splitlines() if l.lower().startswith('content-length')][-1]); self.pos = 0; self.cache = {}; self.nreq = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos
    def _blk(self, i):
        if i not in self.cache:
            a = i * self.B; b = min(self.size, a + self.B) - 1; self.nreq += 1
            self.cache[i] = subprocess.run(['curl', '-s', '-r', f'{a}-{b}', self.url], capture_output=True).stdout
        return self.cache[i]
    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0: return 0
        out = bytearray(); p = self.pos
        while len(out) < n:
            i, o = divmod(p, self.B); chunk = self._blk(i)[o:o + n - len(out)]; out += chunk; p += len(chunk)
        buf[:n] = out; self.pos += n; return n
