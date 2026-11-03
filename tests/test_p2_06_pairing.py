import hashlib
import tempfile
import unittest
from pathlib import Path
from echoproof.data.pairing import _normal, _transcript


class TranscriptTest(unittest.TestCase):
    def test_bom_normalization_and_changed_content(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.txt'; raw=b'\xef\xbb\xbfHello  world\n'; p.write_bytes(raw)
            result=_transcript(p,hashlib.sha256(raw).hexdigest())
            self.assertEqual(result['state'],'ok')
            self.assertTrue(result['utf8_bom'])
            self.assertEqual(result['normalized_sha256'],hashlib.sha256(b'Hello world').hexdigest())
            p.write_bytes(b'changed')
            self.assertEqual(_transcript(p,hashlib.sha256(raw).hexdigest())['state'],'hash_mismatch')

    def test_invalid_empty_and_size_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.txt'
            for data,state in [(b'\xff','invalid_utf8'),(b' \n','empty'),(b'x'*1048577,'resource_limit')]:
                p.write_bytes(data)
                self.assertEqual(_transcript(p,hashlib.sha256(data).hexdigest())['state'],state)
        self.assertEqual(_normal('Ａ   B\n'),'A B')

if __name__=='__main__': unittest.main()
