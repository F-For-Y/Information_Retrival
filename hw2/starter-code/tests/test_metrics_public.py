import unittest
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from relevance import map_score, ndcg_score


class TestRankingMetrics(unittest.TestCase):
    def test_map_score(self):
        # Test calculating Mean Average Precision (MAP)
        results = [1, 1, 0, 1, 0, 0, 0, 1, 0, 0]

        expected_score = 0.6875
        score = map_score(results, 4)

        self.assertAlmostEqual(score, expected_score, places=3,
                               msg=f"Expected MAP score={expected_score}, but got {score}")

    def test_map_score_all_rel(self):
        results = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        expected_score = 1
        score = map_score(results, 4)

        self.assertAlmostEqual(score, expected_score, places=3,
                               msg="MAP does not match expected score")

    def test_map_score_all_rel_missing(self):
        # In this case, we don't have any relevant documents within the cut-off.
        results = [0, 0, 0, 0, 0, 1, 1, 0, 0, 1]

        expected_score = 0
        score = map_score(results, 4)

        self.assertAlmostEqual(score, expected_score, places=3,
                               msg="MAP does not match expected score")

    def test_ndcg_score(self):
        # Test calculating Normalized Discounted Cumulative Gain (NDCG)
        actual = [4, 5, 3, 4, 1, 2, 1, 4, 1, 1]
        ideal = sorted(actual, reverse=True)

        expected_score = 0.953346
        score = ndcg_score(actual, ideal, 4)

        self.assertAlmostEqual(score, expected_score, places=3,
                               msg=f"Expected NDCG score={expected_score}, but got {score}")

    def test_ndcg_ideal_dcg(self):
        # The actual result is already ideal in this case.
        actual = [5, 4, 4, 3, 2, 2, 1, 1, 1, 1]
        ideal = sorted(actual, reverse=True)

        expected_score = 1.0
        score = ndcg_score(actual, ideal, 5)

        self.assertAlmostEqual(score, expected_score, places=3,
                               msg="NDCG does not match expected score")


if __name__ == '__main__':
    unittest.main()
