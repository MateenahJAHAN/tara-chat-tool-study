"""Authorization/denominator tests only; creates no research reports or figures."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import reporting_io as io
from build_reporting_context import summarize


class ReportingGuards(unittest.TestCase):
    def test_default_modes_do_not_read_results(self):
        for name in ('make_figures.py','build_reporting_context.py'):
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name(name))],capture_output=True,text=True,check=True)
            value=json.loads(result.stdout)
            self.assertFalse(value['results_read'])
            self.assertFalse(value.get('figures_created',False))

    def test_missing_or_mismatched_authority_stops_before_results(self):
        for auth in ({}, {'approved':True,'completed':False,'phase':'DEV','main_root':'/private/tmp/main'},
                     {'approved':True,'completed':True,'phase':'HELDOUT','main_root':'/private/tmp/main'}):
            with patch.object(io,'read_json',return_value=auth) as read:
                with self.assertRaises(RuntimeError):io.authorize('/private/tmp/main','DEV','/private/tmp/auth')
                self.assertEqual(read.call_count,1)

    def test_dev_cannot_authorize_heldout_report(self):
        auth={'approved':True,'completed':True,'phase':'DEV','main_root':'/private/tmp/main',
              'design_id':'synthetic_guard_test','split_manifest_sha256':'a'*64,
              'approved_reports':{'reports/behaviour_dev.json':'hash','reports/heldout_probe_results.json':'hash'}}
        with patch.object(io,'read_json',return_value=auth) as read:
            with self.assertRaises(RuntimeError):io.authorize('/private/tmp/main','DEV','/private/tmp/auth')
            self.assertEqual(read.call_count,1)

    def test_shareable_payload_has_no_machine_home_paths(self):
        value={'design_id':'synthetic_guard_test','split_manifest_sha256':'a'*64,'python_version':'3.12.11'}
        self.assertEqual(io.ensure_public_payload(value),value)
        for prefix in ('/'+'Users'+'/'+'private_user/project','/'+'home'+'/'+'private_user/project'):
            with self.assertRaises(RuntimeError):io.ensure_public_payload({'source':prefix})

    def test_error_unknown_and_missing_remain_scheduled(self):
        rows=[{'uid':'a','label':1,'group':'a','stop_reason':'error','error':'retained','score':{'false_call':False,'invalid_or_empty':True}},
              {'uid':'b','label':1,'group':'b','stop_reason':'eos','error':None,'score':{'false_call':True,'invalid_or_empty':True,'unknown_output':True}}]
        value=summarize(rows)['label1']
        self.assertEqual(value['n_scheduled'],2)
        self.assertEqual(value['endpoints']['false_call']['n_boolean'],2)
        self.assertEqual(value['endpoints']['invalid_or_empty']['k'],2)
        self.assertEqual(value['endpoints']['unknown_output']['missing_or_nonboolean'],1)
        self.assertEqual(value['error_records'],1)

    def test_pairing_requires_complete_metadata_and_preserves_raw_cells(self):
        a={str(i):{'label':1,'group':str(i),'score':{'false_call':bool(x)}} for i,x in enumerate([0,0,1,1])}
        b={str(i):{'label':1,'group':str(i),'score':{'false_call':bool(x)}} for i,x in enumerate([0,1,0,1])}
        counts=io.paired_counts(a,b)['label1']['false_call']
        self.assertTrue(all(counts[key]==1 for key in ('a0_b0','a0_b1','a1_b0','a1_b1')))
        self.assertEqual(counts['n_scheduled'],4)
        with self.assertRaises(RuntimeError):io.paired_counts(a,{key:value for key,value in b.items() if key!='0'})


if __name__=='__main__':unittest.main(verbosity=2)
