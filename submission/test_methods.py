"""Scientific invariants, known population structures and leakage boundaries."""
import unittest
import numpy as np
from scipy.sparse.csgraph import connected_components
from network_methods import (correlation, transform_returns, mst, fractional_top_k,
    circular_block_indices, hub_stability, diversification_loss,
    matured_training_rows, covariance_to_correlation, demean_population_covariance)


class ScientificChecks(unittest.TestCase):
    def test_demeaning_covariance_and_correlation_constraints(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(300, 17)) * np.linspace(.4, 2, 17)
        c = transform_returns(x, "demeaned")
        cov = np.cov(c, rowvar=False)
        sigma = c.std(axis=0, ddof=1)
        self.assertLess(np.max(abs(c.sum(axis=1))), 1e-13)
        self.assertLess(np.max(abs(cov @ np.ones(17))), 1e-12)
        self.assertLess(np.max(abs(correlation(c) @ sigma)), 1e-12)

    def test_independent_heteroskedastic_population_formula(self):
        n = 13
        v = np.linspace(.2, 3, n) ** 2
        expected = np.diag(v) - (v[:, None] + v[None, :]) / n + v.sum() / n ** 2
        np.testing.assert_allclose(demean_population_covariance(np.diag(v)), expected, atol=1e-13)
        equal = covariance_to_correlation(demean_population_covariance(np.eye(n)))
        np.testing.assert_allclose(equal.sum(axis=1) - 1, -np.ones(n), atol=1e-12)

    def test_positive_one_factor_population_is_star(self):
        a = np.array([.92, .11, .32, .47, .26, .61, .73])
        r = np.outer(a, a)
        np.fill_diagonal(r, 1)
        for seed in range(10):
            edges, degree = mst(r, np.random.default_rng(seed))
            self.assertEqual(degree[0], len(a) - 1)
            self.assertTrue(np.all(edges[:, 0] == 0))

    def test_tree_connected_and_zero_distance_handled(self):
        r = np.ones((6, 6))
        edges, degree = mst(r, np.random.default_rng(2))
        adjacency = np.zeros_like(r)
        adjacency[edges[:, 0], edges[:, 1]] = 1
        count, _ = connected_components(adjacency + adjacency.T)
        self.assertEqual(count, 1)
        self.assertEqual(len(edges), 5)
        self.assertEqual(degree.sum(), 10)

    def test_fractional_ties_are_label_invariant_and_sum_to_k(self):
        degree = np.array([5, 3, 3, 3, 1])
        expected = np.array([1, 1/3, 1/3, 1/3, 0])
        np.testing.assert_allclose(fractional_top_k(degree, 2), expected)
        perm = np.array([4, 3, 0, 2, 1])
        np.testing.assert_allclose(fractional_top_k(degree[perm], 2), expected[perm])

    def test_blocks_keep_consecutive_joint_observations(self):
        idx = circular_block_indices(252, 10, np.random.default_rng(71))
        self.assertEqual(len(idx), 252)
        for start in range(0, 250, 10):
            np.testing.assert_array_equal(np.diff(idx[start:start+10]) % 252, np.ones(9))

    def test_factor_residuals_orthogonal_with_intercept(self):
        rng = np.random.default_rng(44)
        f = rng.normal(size=(252, 2))
        x = 3 + f @ rng.normal(size=(2, 15)) + rng.normal(size=(252, 15))
        residual = transform_returns(x, "factor_residual", f)
        self.assertLess(np.max(abs(residual.mean(axis=0))), 1e-12)
        self.assertLess(np.max(abs(f.T @ residual)), 1e-10)

    def test_bootstrap_membership_and_deterministic_seed(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(126, 20))
        first = hub_stability(x, None, "raw", 30, 10, 9)
        second = hub_stability(x, None, "raw", 30, 10, 9)
        self.assertAlmostEqual(first.selection.mean(), .1)
        np.testing.assert_array_equal(first.selection, second.selection)

    def test_diversification_target_extremes_and_psd_bound(self):
        rng = np.random.default_rng(61)
        z = rng.normal(size=(200, 1))
        self.assertAlmostEqual(diversification_loss(np.repeat(z, 4, axis=1), np.ones(4)/4), 1)
        self.assertAlmostEqual(diversification_loss(np.column_stack([z, -z]), np.ones(2)/2), 0)
        x = rng.normal(size=(20, 60))
        self.assertTrue(0 <= diversification_loss(x, np.ones(60)/60) <= 1)

    def test_future_label_excluded_until_horizon_matures(self):
        origins = np.array([100, 120, 140, 160])
        np.testing.assert_array_equal(matured_training_rows(origins, 159, 20), [0, 1])
        np.testing.assert_array_equal(matured_training_rows(origins, 160, 20), [0, 1, 2])

    def test_missing_or_constant_data_fail(self):
        with self.assertRaises(ValueError):
            correlation(np.ones((30, 5)))
        x = np.ones((30, 5)); x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            correlation(x)

    def test_reliability_is_a_discount_of_observed_concentration(self):
        rng = np.random.default_rng(124)
        x = rng.normal(size=(126, 20))
        result = hub_stability(x, None, "raw", 20, 10, 76)
        self.assertLessEqual(result.reliability_share, result.degree_concentration + 1e-12)
        self.assertGreaterEqual(result.reliability_share, result.k / (2*(20-1)) - 1e-12)

    def test_forecast_features_and_scaling_do_not_use_future_labels(self):
        import pandas as pd
        from forecast_evaluation import chronological_predictions
        rng = np.random.default_rng(6)
        dates = pd.bdate_range("2007-01-02", periods=2400)[::20]
        frame = pd.DataFrame(dict(origin_date=dates,
            label_end_date=dates+pd.offsets.BDay(20),
            future_y=rng.uniform(.1,.6,len(dates)), feature=rng.normal(size=len(dates))))
        original = chronological_predictions(frame, ["feature"], 1., minimum_training=10)
        changed = frame.copy()
        changed.loc[50:, "future_y"] = .99
        changed.loc[51:, "feature"] = 1000
        second = chronological_predictions(changed, ["feature"], 1., minimum_training=10)
        np.testing.assert_allclose(original.iloc[:51], second.iloc[:51], equal_nan=True)

    def test_asset_specific_sector_factors_are_fitted_per_group(self):
        rng = np.random.default_rng(53)
        factors = rng.normal(size=(126,3))
        tensor = np.empty((126,8,2))
        tensor[:,:4,:] = factors[:,[0,1]][:,None,:]
        tensor[:,4:,:] = factors[:,[0,2]][:,None,:]
        x = 2*tensor[:,:,0] + 3*tensor[:,:,1] + rng.normal(size=(126,8))
        residual = transform_returns(x,"factor_residual",tensor)
        for asset in range(8):
            self.assertLess(np.max(abs(tensor[:,asset,:].T@residual[:,asset])),1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
