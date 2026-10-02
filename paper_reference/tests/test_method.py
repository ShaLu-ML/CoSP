import unittest
import numpy as np
from cosp.method import (prepare_segment, coherence, chronological_split,
                         warning_intervals, select_validation_threshold)


class MethodTests(unittest.TestCase):
    def test_dropout_boundary_and_imputation(self):
        x = np.random.default_rng(42).normal(size=(4000, 16))
        x[:400, 0] = np.nan
        self.assertTrue(np.isfinite(prepare_segment(x, fir_taps=31)).all())
        x[400, 0] = np.nan
        self.assertIsNone(prepare_segment(x, fir_taps=31))

    def test_filter_attenuates_charging_band(self):
        t = np.arange(4000) / 400
        x = np.tile((np.sin(2*np.pi*20*t) + np.sin(2*np.pi*195*t))[:, None], (1, 16))
        y = prepare_segment(x, fir_taps=101)
        center = slice(300, -300)
        expected = np.sin(2*np.pi*20*t)[center]
        self.assertLess(np.std(y[center, 0] - expected), .02)

    def test_coherence_dimensions_pair_order_and_identical_signals(self):
        x = np.random.default_rng(10).normal(size=(4000,16))
        x[:,1] = x[:,0]
        c, f = coherence(x, detrend='constant')
        self.assertEqual(c.shape, (120,109))
        np.testing.assert_allclose(c[0], 1, atol=1e-12)
        self.assertLess(c[1].mean(), .2)
        self.assertEqual(f[0], 0)
        self.assertTrue(f[-1] < 170)

    def test_split_uses_time_not_identifier(self):
        ids = np.arange(10)[::-1]
        a,b,c = chronological_split(ids, np.arange(10))
        self.assertEqual([len(a),len(b),len(c)], [6,2,2])
        np.testing.assert_array_equal(a, ids[:6])
        self.assertFalse(set(a) & set(c))

    def test_probability_average_strict_threshold_retrigger_and_gap(self):
        means, intervals = warning_intervals([10,20,30,40,100,110], [0,1,1,1,1,1],
                                             sop_seconds=20, threshold=.5)
        self.assertEqual(means[1], .5)
        self.assertTrue(np.isnan(means[4]))
        self.assertEqual(intervals, [[30,60],[110,130]])
        with self.assertRaises(ValueError):
            warning_intervals([10,10], [.2,.3], sop_seconds=20, threshold=.5)

    def test_validation_pp_grid(self):
        values = {.2:(1,.8), .5:(.8,.25), .8:(.2,.1)}
        threshold, scores = select_validation_threshold(values, values.__getitem__)
        self.assertEqual(threshold, .5)
        self.assertAlmostEqual(scores[1], .6)

