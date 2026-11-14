"""CPU fixtures; no research inputs or outputs."""
import unittest
import numpy as np
from scipy.optimize import minimize
from run import fit, apply, folds, scores, reliability, loss_arrays, METHODS, EPS, select_threshold


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.p=np.array([.05,.1,.2,.3,.4,.6,.7,.8,.9,.95])
        self.y=np.array([0,0,1,0,1,0,1,1,1,1])

    def test_all_fits_finite_and_monotone(self):
        for m in METHODS:
            v=fit(m,self.y,self.p); z=apply(m,v,self.p)
            self.assertTrue(np.isfinite(z).all()); self.assertTrue((np.diff(z)>=0).all())
            self.assertTrue(((z>=0)&(z<=1)).all())

    def test_platt_constraint(self):
        self.assertGreater(fit('platt',self.y,self.p)['a'],0)

    def test_beta_constraints(self):
        v=fit('beta',self.y,self.p)
        self.assertGreaterEqual(v['a'],0); self.assertGreaterEqual(v['b'],0)

    def test_identity_transform(self):
        for m,v in [('platt',{'a':1,'b':0}),('beta',{'a':1,'b':1,'c':0}),('temperature',{'temperature':1})]:
            np.testing.assert_allclose(apply(m,v,self.p),self.p,atol=1e-15,rtol=0)

    def test_isotonic_nll_solution(self):
        v=fit('isotonic',self.y,self.p); z=apply('isotonic',v,self.p)
        opt=minimize(lambda p:loss_arrays(self.y,p)[:,0].mean(), np.linspace(.1,.9,10),
                     method='SLSQP',bounds=[(EPS,1-EPS)]*10,
                     constraints={'type':'ineq','fun':lambda p:np.diff(p)},options={'ftol':1e-10,'maxiter':2000})
        self.assertTrue(opt.success)
        self.assertAlmostEqual(loss_arrays(self.y,z)[:,0].mean(),opt.fun,places=6)

    def test_group_design(self):
        for ng,k,size in [(12,4,3),(25,5,5)]:
            g=np.repeat([f's{i:02}' for i in range(ng)],4)
            for seed in (42,43,44):
                f=folds(g,k,seed)
                self.assertEqual([len(x) for x in f],[size]*k)
                self.assertEqual(len(set(np.concatenate(f))),ng)
                self.assertEqual(np.concatenate(f).tolist(),np.concatenate(folds(g,k,seed)).tolist())
        self.assertEqual(len(folds(np.repeat(np.arange(12),3))),12)

    def test_class_missing_rejected(self):
        with self.assertRaises(ValueError):fit('platt',np.zeros(10),self.p)

    def test_contract_threshold_python_scalar(self):
        for m in METHODS:
            z=apply(m,fit(m,self.y,self.p),self.p)
            rule,_,_=select_threshold(self.y.tolist(),z.tolist())
            self.assertEqual(len([rule.decide(float(x)) for x in z]),len(z))

    def test_ece_edges(self):
        self.assertAlmostEqual(reliability(np.array([0,1]),np.array([.1,.9]))[0],.1)
        self.assertEqual(reliability(np.array([0,1]),np.array([0.,1.]))[0],0)

    def test_equal_count_counts(self):
        _,r=reliability(np.tile([0,1],11),np.linspace(0,1,22),True)
        self.assertEqual(sum(x['count'] for x in r),22)
        self.assertLessEqual(max(x['count'] for x in r)-min(x['count'] for x in r),1)

    def test_rank_preserved_single_fit(self):
        for m in ['temperature','platt','beta']:
            z=apply(m,fit(m,self.y,self.p),self.p)
            self.assertAlmostEqual(scores(self.y,z)['auroc'],scores(self.y,self.p)['auroc'])


if __name__=='__main__':unittest.main()
