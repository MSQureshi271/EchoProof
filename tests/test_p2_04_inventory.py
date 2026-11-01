import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from echoproof.data.inventory import run_inventory


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {'sources':{},'outputs':{'inventory':'output'},'block_bytes':3,'retries':2,'checkpoint_interval':1,'immutable_inputs':False}
        for name in ('genuine','transcripts','xtts','chatterbox','melo','vits'):
            self.config['sources'][name] = 'source/'+name
            (self.root/'source'/name).mkdir(parents=True)
        (self.root/'source/genuine/a.wav').write_bytes(b'abc')
        (self.root/'source/xtts/a.wav').write_bytes(b'abc')
        (self.root/'source/transcripts/a.txt').write_bytes(b'text')
        (self.root/'source/chatterbox/empty.wav').write_bytes(b'')
        (self.root/'source/melo/a.wav').write_bytes(b'melo')
        (self.root/'source/vits/a.wav').write_bytes(b'vits')
        self.cfg = self.root/'config'; self.cfg.write_text('config')
        self.registry = self.root/'registry'; self.registry.write_text('registry')

    def tearDown(self): self.temp.cleanup()

    def run_scan(self, **kw):
        return run_inventory(self.root,self.config,self.cfg,self.registry,**kw)

    def test_resume_equals_fresh_and_preserves_duplicates(self):
        stopped = self.run_scan(stop_after=2)
        resumed = self.run_scan(resume=stopped['run_id'])
        fresh = self.run_scan()
        self.assertEqual(resumed['content_digest_sha256'],fresh['content_digest_sha256'])
        path = self.root/'output'/resumed['run_id']/'file_inventory.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows),6)
        self.assertEqual({r['component'] for r in rows}, {'genuine','transcripts','xtts','chatterbox','melo','vits'})
        equal = [r for r in rows if r['sha256'] == hashlib.sha256(b'abc').hexdigest()]
        self.assertEqual(len(equal),2, rows)
        self.assertNotEqual(equal[0]['file_id'],equal[1]['file_id'])
        self.assertEqual((self.root/'source/genuine/a.wav').read_bytes(),b'abc')

    def test_changed_file_rehashed_and_tamper_rejected(self):
        stopped = self.run_scan(stop_after=2)
        (self.root/'source/genuine/a.wav').write_bytes(b'changed')
        result = self.run_scan(resume=stopped['run_id'])
        self.assertEqual(result['content_digest_sha256'],self.run_scan()['content_digest_sha256'])
        db = sqlite3.connect(self.root/'output'/stopped['run_id']/'checkpoint.sqlite')
        db.execute("UPDATE rows SET row_json='{}' WHERE component='genuine'"); db.commit(); db.close()
        with self.assertRaisesRegex(ValueError,'checksum'): self.run_scan(resume=stopped['run_id'])

    def test_missing_source_and_config_change(self):
        (self.root/'source/xtts/a.wav').unlink()
        (self.root/'source/xtts').rmdir()
        result = self.run_scan()
        self.assertEqual(result['status'],'needs_attention')
        self.assertEqual(result['counts']['error'],1)
        self.cfg.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'): self.run_scan(resume=result['run_id'])


if __name__ == '__main__': unittest.main()
