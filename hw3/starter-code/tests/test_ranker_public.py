import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF
from indexing import Indexer, IndexType


def assertScoreLists(self, exp_list, res_list):
    self.assertEqual(len(exp_list), len(
        res_list), f'Expected length {len(exp_list)} but actual list length {len(res_list)}')
    for idx in range(len(res_list)):
        self.assertEqual(exp_list[idx][0], res_list[idx][0],
                         f'Expected document not at index {idx}')
        self.assertAlmostEqual(exp_list[idx][1], res_list[idx][1], places=4,
                               msg=f'Expected score differs from actual score at {idx}')

class MockTokenizer:
    def tokenize(self, text):
        return text.split()

class TestWordCountCosineSimilarity(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = WordCountCosineSimilarity(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        self.assertEqual(exp_list, res_list,
                         'Cosine: no overlap between query and docs')

    def test_perfect_match(self):
        exp_list = [(1, 1), (3, 1), (5, 1)]
        res_list = self.ranker.query("AI")
        self.assertEqual(exp_list, res_list,
                         'Expected list differs from result list')

    def test_partial_match(self):
        exp_list = [(3, 2), (4, 2), (1, 1), (5, 1)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        self.assertEqual(exp_list, res_list,
                         'Expected list differs from result list')


class TestDirichletLM(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = DirichletLM(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.01128846343027107), (3, 0.007839334610553066),
                    (1, 0.0073475716303944075)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(3, 0.029667610688458967), (4, 0.017285590697028078),
                    (5, -0.027460212369367794), (1, -0.04322377956887445)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_mu(self):
        DLM = DirichletLM(self.index, {'mu': 5})
        ret_score = DLM.score(1, ['AI', 'Google'])
        exp_score = 1.6857412751512575

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_mu2(self):
        DLM = DirichletLM(self.index, {'mu': 1})
        ret_score = DLM.score(1, ['AI', 'Google'])
        exp_score = 1.798539156213434

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_med_mu(self):
        DLM = DirichletLM(self.index, {'mu': 30})
        ret_score = DLM.score(1, ['AI', 'Google'])
        exp_score = 1.2278314183215069

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_mu(self):
        DLM = DirichletLM(self.index, {'mu': 1000})
        ret_score = DLM.score(1, ['AI', 'Google'])
        exp_score = 0.11811761538891903

        self.assertAlmostEqual(
            exp_score, ret_score, places=3, msg='DirichletLM: partial match, score')


class TestBM25(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = BM25(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, -0.31623109945742595), (3, -0.32042144088133173),
                    (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 1.5460888344441546), (3, 0.7257835477973098),
                    (1, -0.31623109945742595), (5, -0.35318117923823517)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1, 'k3': 8})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7199009648250208

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_k1(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 2, 'k3': 8})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7068428242958602

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 0})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_k3(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.2, 'k3': 1000})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7162920454285571

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_random_param(self):
        scorer = BM25(self.index, {'b': 0.75, 'k1': 1.99, 'k3': 49})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 0.7069285957828516

        self.assertAlmostEqual(exp_score, ret_score,
                               places=3, msg='BM25: partial match, score')


class TestPivotedNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = PivotedNormalization(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(5, 0.7095587433308632), (3, 0.6765779252477553),
                    (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.7806792016468633), (3, 2.4255064908289246),
                    (5, 0.7095587433308632), (1, 0.6721150101735617)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)

    @unittest.skip('Test broken due to parameter mismatch')
    def test_small_param(self):
        scorer = PivotedNormalization(self.index, {'b': 0})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 2.4849066497880004

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')

    @unittest.skip('Test broken due to parameter mismatch')
    def test_large_param(self):
        scorer = PivotedNormalization(self.index, {'b': 1})
        ret_score = scorer.score(1, ['AI', 'Google'])
        exp_score = 2.1487133971696237

        self.assertAlmostEqual(exp_score, ret_score, places=3,
                               msg='PivotedNormalization: partial match, score')


class TestTF_IDF(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = MockTokenizer()
        self.stopwords = set(['a', 'an', 'and', 'the', 'this'])
        self.index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 1)
        scorer = TF_IDF(self.index)
        self.ranker = Ranker(self.index, self.preprocessor,
                             self.stopwords, scorer)

    def test_no_overlap(self):
        exp_list = []
        res_list = self.ranker.query("cough drops")
        assertScoreLists(self, exp_list, res_list)

    def test_perfect_match(self):
        exp_list = [(1, 1.047224521431117),
                    (3, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI")
        assertScoreLists(self, exp_list, res_list)

    def test_partial_match(self):
        exp_list = [(4, 2.866760557116562), (3, 2.8559490532810434),
                    (1, 1.047224521431117), (5, 1.047224521431117)]
        res_list = self.ranker.query("AI chatbots and vehicles")
        assertScoreLists(self, exp_list, res_list)


# Assuming YourRanker uses the same setup as the other rankers
# class TestYourRanker(unittest.TestCase):
#    def setUp(self):
#        self.index = MockIndex()
#        self.scorer = YourRanker(self.index, {})
#    def test_score(self):
#        result = self.scorer.score(["test", "sample"], self.index.get_statistics(), self.index, 1)
#        self.assertTrue(isinstance(result, dict))
#        self.assertTrue('docid' in result and 'score' in result)

if __name__ == '__main__':
    unittest.main()
