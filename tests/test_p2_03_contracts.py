import json
import tempfile
import unittest
from pathlib import Path

from echoproof.contracts import ContractError
from echoproof.data.audit_contracts import (FileInventory, Provenance, MappingEvidence,
    PairRelationship, Disposition, Reconciliation, validate_scope)


class AuditContractsTest(unittest.TestCase):
    def test_roundtrip_unknown_and_evidence(self):
        rows = [
            FileInventory(1,'research','phonemedf:f1','phonemedf','genuine','a.wav','wav',None,'2026-10-30',None,None,None,None,'pending'),
            Provenance(1,'research','phonemedf','phonemedf:snapshot',None,'user_attested',None,None,['doc'],[],None),
            MappingEvidence(1,'research','phonemedf:f1','language',None,None,False,None),
            PairRelationship(1,'research','phonemedf:f1','phonemedf:f2','same_utterance',None,'candidate',None,None),
            Disposition(1,'research','phonemedf:f1','awaiting_review','admission','pending',None,'2026-10-30'),
            Reconciliation(1,'research','phonemedf',None,None,None,None,'inventory',None,None,'not run')]
        for row in rows:
            self.assertEqual(type(row).from_json(row.to_json()), row)
        validate_scope(rows, 'research')
        with self.assertRaises(ContractError): validate_scope(rows, 'fixture')
        with self.assertRaises(ContractError): MappingEvidence(1,'research','fixture:f1','label',None,None,False,None)
        with self.assertRaises(ContractError): MappingEvidence(1,'research','phonemedf:f1','label',None,None,True,None)
        with self.assertRaises(ContractError): FileInventory(1,'research','phonemedf:f1','s','c','../x','wav',0,'now',None,None,None,None,'completed')


try:
    from echoproof.phase2_config import load_phase2_config, require_acquisition_capacity
    from echoproof.config import ConfigError
except ModuleNotFoundError as exc:
    if exc.name != 'yaml': raise
    load_phase2_config = None


@unittest.skipIf(load_phase2_config is None, 'PyYAML unavailable')
class AuditConfigTest(unittest.TestCase):
    def test_invalid_configs_and_capacity(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'configs/phase2.yaml').read_text(encoding='utf-8')
        config = load_phase2_config(root / 'configs/phase2.yaml',root)
        with self.assertRaises(ConfigError): require_acquisition_capacity(config, 1)
        variants = [source + '\nworkers: 2\n', source + '\nunknown: true\n',
            source.replace('workers: 1','workers: true'),
            source.replace('dataset/genuine/wav','../outside'),
            source.replace('data/manifests/phase2','dataset/genuine'),
            source.replace('data/staging/phase2','configs')]
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / 'bad.yaml'
            for variant in variants:
                path.write_text(variant,encoding='utf-8')
                with self.subTest(variant=variant), self.assertRaises(ConfigError):
                    load_phase2_config(path, root)


if __name__ == '__main__': unittest.main()
