import unittest
import json
from collections import Counter, defaultdict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# modules being tested
from document_preprocessor import RegexTokenizer
from indexing import IndexType, Indexer
from ranker import BM25, PivotedNormalization, TF_IDF, DirichletLM, WordCountCosineSimilarity


class TestRankingMetrics(unittest.TestCase):

    def get_doc_counts(self, dataset_name):
        docid_to_word_counts = defaultdict(Counter)
        rt = RegexTokenizer('\\w+')
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = rt.tokenize(d['text'])
                docid_to_word_counts[docid] = Counter(tokens)

        return docid_to_word_counts

    def test_bm25_single_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            1: -1.0478492565048942,
            3: -1.0216238857914792,
            5: -1.151287838704531,
        }
        bm25 = BM25(index)

        for docid, expected_score in docid_to_score.items():
            score = bm25.score(docid, docid_to_word_counts[docid], Counter(['ai']))
            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_bm25_multi_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            4: 0.4096166920058151,
            3: 0.0,
            1: -1.0478492565048942,
            5: -1.151287838704531,
        }
        bm25 = BM25(index)

        for docid, expected_score in docid_to_score.items():
            score = bm25.score(docid, docid_to_word_counts[docid], Counter([
                               'ai', 'chatbots', 'vehicles']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)



    def test_tf_idf_single_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            1: 0.8478185040135232,
            3: 0.8478185040135232,
            5: 0.8478185040135232
        }
        tfidf = TF_IDF(index)

        for docid, expected_score in docid_to_score.items():
            score = tfidf.score(docid, docid_to_word_counts[docid], Counter(['ai']))
#                print(docid)
#                print(score)
            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_tf_idf_multi_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            4: 3.714579061130085,
            3: 2.656543035863449,
            1: 0.8478185040135232,
            5: 0.8478185040135232,
        }
        tfidf = TF_IDF(index)

        for docid, expected_score in docid_to_score.items():
            score = tfidf.score(docid, docid_to_word_counts[docid], Counter([
                                'ai', 'chatbots',  'vehicles']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_pivoted_normalization_single_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            5: 0.4147422370958343,
            3: 0.39105771847995546,
            1: 0.3960841673036053,
        }
        pn = PivotedNormalization(index)

        for docid, expected_score in docid_to_score.items():
            score = pn.score(docid, docid_to_word_counts[docid], Counter(['ai']))
#                print(docid)
#                print(score)
            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_pivoted_normalization_multi_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            4: 3.191131756539232,
            3: 2.119150607583155,
            5: 0.4147422370958343,
            1: 0.3960841673036053,
        }
        pn = PivotedNormalization(index)

        for docid, expected_score in docid_to_score.items():
            score = pn.score(docid, docid_to_word_counts[docid], Counter([
                             'ai', 'chatbots', 'vehicles']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_word_count_cosine_similarity_single_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            4: 1,
            3: 1,
            5: 1,
            1: 1
        }
        wccs = WordCountCosineSimilarity(index)

        for docid, expected_score in docid_to_score.items():
            score = wccs.score(docid, docid_to_word_counts[docid], Counter(['ai']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_word_count_cosine_similarity_multi_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            4: 3,
            3: 2,
            5: 1,
            1: 1
        }
        wccs = WordCountCosineSimilarity(index)

        for docid, expected_score in docid_to_score.items():
            score = wccs.score(docid, docid_to_word_counts[docid], Counter([
                               'ai', 'chatbots', 'vehicles']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_dirichlet_lm_single_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            5: 0.00541207,
            3: 0.00098184,
            1: 0.00196464
        }
        dlm = DirichletLM(index)

        for docid, expected_score in docid_to_score.items():
            score = dlm.score(docid, docid_to_word_counts[docid], Counter(['ai']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)

    def test_dirichlet_lm_multi_word_query(self):

        index = Indexer.create_index(
            IndexType.InvertedIndex, 'dataset_1.jsonl', RegexTokenizer('\\w+'), set(), 0)
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        docid_to_score = {
            3: 0.03855246,
            4: 0.0503635,
            5: -0.02140731,
            1: -0.0317496
        }
        dlm = DirichletLM(index)

        for docid, expected_score in docid_to_score.items():
            score = dlm.score(docid, docid_to_word_counts[docid], Counter([
                              'ai', 'chatbots', 'vehicles']))

            self.assertAlmostEqual(score, expected_score,
                                   msg='Wrong score for docid %d, expected %f' % (
                                       docid, expected_score),
                                   places=3)


if __name__ == '__main__':
    unittest.main()