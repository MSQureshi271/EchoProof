import unittest
from echoproof.contracts import ContractError
from echoproof.data.phonemedf_adapter import adapt_row


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.config={'sources':{k:'dataset/'+k for k in ('genuine','xtts','chatterbox','melo','vits')}}

    def row(self, component, identity):
        return {'component':component,'file_id':'phonemedf:'+identity,'state':'completed',
                'file_type':'.wav','relative_path':'19-198-0000.wav','sha256':'a'*64}

    def test_each_family_provisional_and_ids_distinct(self):
        ids=set()
        for component in self.config['sources']:
            row=self.row(component,component)
            record, disposition=adapt_row(row,self.config,'phonemedf:snapshot')
            self.assertIsNotNone(record)
            self.assertFalse(record.label_verified)
            self.assertFalse(record.method_verified)
            self.assertEqual(disposition['state'],'pending')
            with self.assertRaises(ContractError): record.validate_supervised()
            self.assertEqual(record,adapt_row(row,self.config,'phonemedf:snapshot')[0])
            ids.add(record.recording_id)
        self.assertEqual(len(ids),5)

    def test_unknown_unreadable_and_bad_hash_cannot_be_admitted(self):
        row=self.row('voice_conversion','x')
        self.assertIsNone(adapt_row(row,self.config,'phonemedf:snapshot')[0])
        row=self.row('xtts','x'); row['sha256']='invalid'
        self.assertEqual(adapt_row(row,self.config,'phonemedf:snapshot')[1]['reason'],'invalid_record')
        row=self.row('xtts','x'); row['relative_path']='../escape.wav'
        self.assertEqual(adapt_row(row,self.config,'phonemedf:snapshot')[1]['reason'],'invalid_record')
        row=self.row('xtts','x'); row['state']='error'
        self.assertIsNone(adapt_row(row,self.config,'phonemedf:snapshot')[0])

if __name__=='__main__': unittest.main()
