"""Small software fixtures; no model imports or research execution."""
import csv
from pathlib import Path
import tempfile
import unittest
from verify_handoff import checked_path, sha, verify_files, verify_ledger


class HandoffTests(unittest.TestCase):
    def test_exact_file_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'checkpoint'; p.write_bytes(b'fixture')
            identity = {'checkpoint': {'bytes': p.stat().st_size, 'sha256': sha(p)}}
            verify_files(temp, identity)
            p.write_bytes(b'changed')  # equal size: hash must detect corruption
            with self.assertRaises(ValueError): verify_files(temp, identity)

    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                verify_files(temp, {'missing': {'bytes': 0, 'sha256': 'bad'}})

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in ('../outside', str(Path(temp).resolve())):
                with self.subTest(path=path), self.assertRaises(ValueError): checked_path(temp, path)

    def test_ledger_binds_row_without_freezing_unrelated_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'ledger.csv'
            row = {'run_id': 'candidate', 'status': 'completed'}
            contract = {'locations': ['ledger.csv'], 'row': row}
            def write(rows):
                with p.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            write([row, {'run_id': 'teammate', 'status': 'completed'}]); verify_ledger(temp, contract)
            write([row, row])
            with self.assertRaises(ValueError): verify_ledger(temp, contract)
            write([{'run_id': 'candidate', 'status': 'failed'}])
            with self.assertRaises(ValueError): verify_ledger(temp, contract)


if __name__ == '__main__':
    unittest.main()
