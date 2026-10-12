"""Seekable HTTP-range reader for the official concatenated EvalMuse ZIP parts.

Pin repository revision; validate every Content-Range and byte count. ZIP CRC32
checks extracted members; whole-archive LFS SHA256 is NOT verified by partial reads.
"""
import io
import re
import subprocess
from pathlib import Path


class RangeZip(io.RawIOBase):
    def __init__(self, parts, base_url, cache, block_size=2*1024*1024):
        self.parts = parts
        self.base_url = base_url
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.block_size = block_size
        self.position = 0
        self.size = sum(item['size'] for item in parts)
        self.downloaded_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def _block(self, item, start):
        end = min(item['size']-1, start+self.block_size-1)
        path = self.cache/(item['rfilename']+'.'+str(start))
        if path.exists() and path.stat().st_size == end-start+1:
            return path.read_bytes()
        header = path.with_suffix(path.suffix+'.headers')
        temporary = path.with_suffix(path.suffix+'.tmp')
        url = self.base_url+'/'+item['rfilename']+'?range='+str(start)+'-'+str(end)
        for attempt in range(3):
            try:
                subprocess.run(['/usr/bin/curl', '-fL', '--http1.1', '--connect-timeout', '20',
                    '--max-time', '120', '--max-filesize', str(self.block_size),
                    '-r', str(start)+'-'+str(end), '-D', str(header), url, '-o', str(temporary)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ranges = re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)', header.read_text().lower())
                if not ranges or tuple(map(int, ranges[-1])) != (start, end, item['size']):
                    raise ValueError('Unexpected server range')
                if temporary.stat().st_size != end-start+1:
                    raise ValueError('Incomplete HTTP range')
                temporary.replace(path)
                self.downloaded_bytes += end-start+1
                return path.read_bytes()
            except (subprocess.CalledProcessError, ValueError):
                if attempt == 2:
                    raise

    def read(self, length=-1):
        length = min(self.size-self.position, length if length >= 0 else self.size-self.position)
        if length <= 0:
            return b''
        if length > 64*1024*1024:
            raise ValueError('Refusing large unbounded ZIP read')
        chunks = []
        while length:
            base = 0
            for item in self.parts:
                if self.position < base+item['size']:
                    break
                base += item['size']
            offset = self.position-base
            start = offset//self.block_size*self.block_size
            block = self._block(item, start)
            local = offset-start
            take = min(length, len(block)-local)
            chunks.append(block[local:local+take])
            self.position += take
            length -= take
        return b''.join(chunks)
