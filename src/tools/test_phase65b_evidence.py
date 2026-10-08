"""Check paired inference and cohort/denominator rejection without ML runtime."""
import ast
from pathlib import Path
import unittest
import numpy as np


def functions(path,names):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]


ns={'np':np}
definitions=functions('src/tools/summarize_phase65a_explore.py',{'pairwise_auroc'})+functions('src/tools/phase65b_evidence_probe.py',{'pair_rows','paired_inference'})
exec(compile(ast.Module(body=definitions,type_ignores=[]),'<protocol functions>','exec'),ns)


class EvidenceTests(unittest.TestCase):
    def test_same_scores_have_no_increment(self):
        y=np.array([True]*8+[False]*8);p=np.linspace(.1,.9,16)
        r=ns['paired_inference'](y,p,p,iterations=30)
        self.assertEqual(r['paired_stratified_ci95'],[0,0]);self.assertEqual(r['holm_family3_adjusted_p'],1)

    def test_valid_pairs_and_denominator_mismatch(self):
        row=dict(product='x',group_id='g',cohort='fit_representatives',confusion=[[3,0,0],[0,3,0],[1,0,2]],source_observable_features=[.1]*12,miou_percent=50)
        a=dict(row);a['confusion']=[[3,0,0],[0,3,0],[2,0,1]]
        r=ns['pair_rows']({'scenes':[row]},{'scenes':[a]},'fit_representatives')
        self.assertTrue(r[0]['risk'])
        a['confusion']=[[3,0,0],[0,3,0],[2,0,0]]
        with self.assertRaises(ValueError):ns['pair_rows']({'scenes':[row]},{'scenes':[a]},'fit_representatives')

    def test_repeated_group_rejected(self):
        base=dict(group_id='g',cohort='c',confusion=np.eye(3,dtype=int).tolist(),source_observable_features=[.1]*12,miou_percent=50)
        rows=[dict(base,product='x'),dict(base,product='y')]
        with self.assertRaises(ValueError):ns['pair_rows']({'scenes':rows},{'scenes':rows},'c')


if __name__=='__main__':unittest.main()
