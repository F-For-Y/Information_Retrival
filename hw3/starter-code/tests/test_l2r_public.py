import unittest
import json
from collections import Counter, defaultdict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from document_preprocessor import RegexTokenizer
from indexing import IndexType, Indexer
from ranker import Ranker, BM25
# modules being tested
from l2r import LambdaMART, L2RFeatureExtractor, L2RRanker
from ranker import CrossEncoderScorer


class TestL2RRanker(unittest.TestCase):

    def setUp(self) -> None:
        self.preprocessor = RegexTokenizer('\w+')

        self.stopwords = set()
        self.doc_category_info = dict()

        with open('stopwords.txt', 'r', encoding='utf-8') as file:
            for stopword in file:
                self.stopwords.add(stopword)

        self.main_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 0)
        self.title_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 0, text_key='title')
        self.network_features = {
            1: {'pagerank': 0.1, 'hub_score': 0.11, 'authority_score': 0.111},
            2: {'pagerank': 0.2, 'hub_score': 0.22, 'authority_score': 0.222},
            3: {'pagerank': 0.3, 'hub_score': 0.33, 'authority_score': 0.333},
            4: {'pagerank': 0.4, 'hub_score': 0.44, 'authority_score': 0.444},
            5: {'pagerank': 0.5, 'hub_score': 0.55, 'authority_score': 0.555},
        }

        raw_text_dict = {}
        with open('dataset_1.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                raw_text_dict[int(data['docid'])] = data['text'][:500]

        self.ce_scorer = CrossEncoderScorer(raw_text_dict)

        self.recognized_categories = set(['Category 100', 'Category 11'])

        # Create a dictionary where each document is mapped to its list of categories.
        with open("dataset_1.jsonl", 'r') as f:
            for line in f:
                d = json.loads(line)
                docid = d["docid"]
                self.doc_category_info[docid] = d["categories"]

        self.ranker = Ranker(self.main_index, self.preprocessor,
                             self.stopwords, BM25(self.main_index))

        self.fe = L2RFeatureExtractor(self.main_index,
                                      self.title_index,
                                      self.doc_category_info,
                                      self.preprocessor,
                                      self.stopwords,
                                      self.recognized_categories,
                                      self.network_features,
                                      self.ce_scorer)

    def test_init(self):
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe)

        self.assertIsNotNone(l2r, "Failed in __init__ of L2RRanker")

    def test_prepare_training_data(self):
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe)

        query_to_doc_rels = {
            "ai": [(1, 1), (2, 3), (4, 1)],
            "ai ml": [(1, 5), (2, 1), (3, 1), (4, 2)],
        }

        X, y, qrels = l2r.prepare_training_data(query_to_doc_rels)

        self.assertEqual(
            len(X), 7, "Expected 7 feature vectors, i.e., len(X) from prepare_training_data")
        self.assertEqual(
            len(y), 7, "Expected 7 relevance scores, i.e., len(y) from prepare_training_data")
        self.assertEqual(len(
            qrels), 2, "Expected 2 query group sizes, i.e., len(qrels) from prepare_training_data")
        self.assertEqual(
            qrels[0], 3, "Expected first query group to have 3 relevance scores")
        self.assertEqual(
            qrels[1], 4, "Expected first query group to have 3 relevance scores")

    def test_predict(self):
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe)

        with self.assertRaises(ValueError):
            # Expecting a ValueError since model is not trained yet
            l2r.predict([])

        l2r.train('data-relevance.csv')

        for i in range(1, 20):
            temp_feature_lst = list(range(1, i + 1))
            try:
                l2r.predict([temp_feature_lst])
                feature_lst = temp_feature_lst
            except Exception as e:
                print(e)
                continue

        # Number of features used
        print(f"Number of features used: {len(feature_lst)}")

        predictions = l2r.predict([feature_lst, feature_lst])

        # Assume these are valid feature vectors
        self.assertEqual(len(predictions), 2)

    def test_query_with_valid_single_term_query(self):
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe)
        l2r.train('data-relevance.csv')
        result = l2r.query("chatbots")
        self.assertIsInstance(result, list)
        self.assertTrue(
            result, "Expected non-empty list of results for query 'chatbot'")

    def test_query_with_valid_multi_term_query(self):
        l2r = L2RRanker(self.main_index, self.title_index, self.preprocessor,
                        self.stopwords, self.ranker, self.fe)
        l2r.train('data-relevance.csv')
        result = l2r.query("AI ML")  # Assume this is a valid multi-term query
        self.assertIsInstance(result, list)
        self.assertTrue(
            result, "Expected non-empty list of results for query 'AI ML'")


class TestL2RFeatureExtractor(unittest.TestCase):
    def setUp(self) -> None:
        self.preprocessor = RegexTokenizer('\w+')

        self.stopwords = set()
        self.doc_category_info = dict()

        with open('stopwords.txt', 'r', encoding='utf-8') as file:
            for stopword in file:
                self.stopwords.add(stopword)

        raw_text_dict = {}
        with open('dataset_1.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                raw_text_dict[int(data['docid'])] = data['text'][:500]

        self.ce_scorer = CrossEncoderScorer(raw_text_dict)

        self.main_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 0)
        self.title_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, 'dataset_1.jsonl', self.preprocessor, self.stopwords, 0, text_key='title')
        self.network_features = {
            1: {
                'pagerank': 0.1,
                'hub_score': 0.11,
                'authority_score': 0.111
            },
            2: {
                'pagerank': 0.2,
                'hub_score': 0.22,
                'authority_score': 0.222
            },
            3: {
                'pagerank': 0.3,
                'hub_score': 0.33,
                'authority_score': 0.333
            },
            4: {
                'pagerank': 0.4,
                'hub_score': 0.44,
                'authority_score': 0.444
            },
            5: {
                'pagerank': 0.5,
                'hub_score': 0.55,
                'authority_score': 0.555
            }
        }

        self.recognized_categories = set(['Category 100', 'Category 11'])

        # Create a dictionary where each document is mapped to its list of categories.
        with open("dataset_1.jsonl", 'r') as f:
            for line in f:
                d = json.loads(line)
                docid = d["docid"]
                self.doc_category_info[docid] = d["categories"]

    def get_doc_counts(self, dataset_name):
        docid_to_word_counts = defaultdict(Counter)
        with open(dataset_name) as f:
            for line in f:
                d = json.loads(line)
                docid = d['docid']
                tokens = self.preprocessor.tokenize(d['text'])
                docid_to_word_counts[docid] = Counter(tokens)

        return docid_to_word_counts

    def test_get_article_length(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        docid_to_len = {
            1: 34, 2: 27, 3: 36, 4: 28, 5: 27
        }

        for docid, doclen in docid_to_len.items():
            dlen = fe.get_article_length(docid)
            self.assertEqual(dlen, doclen, "Expected len=%d for docid %d, but got %d" % (
                doclen, docid, dlen))

    def test_get_tf(self):

        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        docid_to_tf = {
            1: 0.693147, 2: 0, 3: 0.693147, 4: 0.693147, 5: 0.693147
        }

        for docid, tf in docid_to_tf.items():
            est_tf = fe.get_tf(self.main_index, docid,
                               docid_to_word_counts[docid], ['ai'])
            self.assertAlmostEqual(tf, est_tf, places=3,
                                   msg="Expected tf=%f for docid %d, but got %f" % (tf, docid, est_tf))

    def test_get_tfidf(self):

        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        docid_to_tfidf = {
            1: 0.8478185040135232, 2: 0, 3: 0.8478185040135232, 4: 0.8478185040135232, 5: 0.8478185040135232
        }

        for docid, tf_idf in docid_to_tfidf.items():
            est_tf_idf = fe.get_tf_idf(
                self.main_index, docid, docid_to_word_counts[docid], ['ai'])
            # print(est_tf_idf)
            self.assertAlmostEqual(est_tf_idf, tf_idf, places=3,
                                   msg=f"Expected tf-idf={tf_idf} for docid {docid}, but got {est_tf_idf}")

    def test_get_BM25_score(self):
        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        docid_to_BM25 = {
            1: -1.047849, 2: 0, 3: -1.021624, 4: -1.135278, 5: -1.1512878
        }

        for docid, BM25_score in docid_to_BM25.items():
            est_BM25 = fe.get_BM25_score(
                docid, docid_to_word_counts[docid], ['ai'])
            self.assertAlmostEqual(est_BM25, BM25_score, places=3,
                                   msg=f"Expected bm25={BM25_score} for docid {docid}, but got {est_BM25}")

    def test_get_pivoted_normalization_score(self):

        docid_to_word_counts = self.get_doc_counts('dataset_1.jsonl')

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        docid_to_pn = {
            1: 0.396084, 2: 0, 3: 0.3910577, 4: 0.4119699, 5: 0.4147422
        }

        for docid, pn_score in docid_to_pn.items():
            est_pn = fe.get_pivoted_normalization_score(
                docid, docid_to_word_counts[docid], ['ai'])
            self.assertAlmostEqual(est_pn, pn_score, places=3,
                                   msg=f"Expected pivoted normalization score={pn_score} for docid {docid}, but got {est_pn}")

    def test_get_pagerank(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        for docid, network_stats in self.network_features.items():
            actual_pagerank = fe.get_pagerank_score(docid)
            expected_pagerank = network_stats['pagerank']
            self.assertEqual(actual_pagerank, expected_pagerank,
                             msg=f'Expected pagerank={expected_pagerank} but got {actual_pagerank} for document {docid}')

    def test_get_hub(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        for docid, network_stats in self.network_features.items():
            actual_hub = fe.get_hits_hub_score(docid)
            expected_hub = network_stats['hub_score']
            self.assertEqual(actual_hub, expected_hub,
                             msg=f'Expected HITS hub score={expected_hub} but got {actual_hub} for document {docid}')

    def test_get_authority(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        for docid, network_stats in self.network_features.items():
            actual_auth = fe.get_hits_authority_score(docid)
            expected_auth = network_stats['authority_score']
            self.assertEqual(actual_auth, expected_auth,
                             msg=f'Expected HITS hub score={expected_auth} but got {actual_auth} for document {docid}')

    def test_get_categories(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        # We can't really guarantee the ordering but we can at least sum to test that the
        # right number of categories were recognized
        expected_lst = [[0, 1], [0, 1], [1, 1], [0, 0], [1, 1]]

        for docid in [1, 2, 3, 4, 5]:
            actual_vector = fe.get_document_categories(docid)
            expected_vector = expected_lst[docid - 1]
            self.assertEqual(sum(actual_vector), sum(expected_vector),
                             "Expected to see %d categories but saw %d"
                             % (sum(actual_vector), sum(expected_vector)))

    def test_get_cross_encoder_score(self):

        fe = L2RFeatureExtractor(self.main_index,
                                 self.title_index,
                                 self.doc_category_info,
                                 self.preprocessor,
                                 self.stopwords,
                                 self.recognized_categories,
                                 self.network_features,
                                 self.ce_scorer)

        expected_lst = [-4.282045841217041, -11.096477508544922, 1.6918237209320068, 4.874917030334473, -6.669098377227783]

        for docid in [1, 2, 3, 4, 5]:
            actual_score = fe.get_cross_encoder_score(docid, 'AI chatbots and vehicles')
            expected_score = expected_lst[docid - 1]
            self.assertAlmostEqual(actual_score, expected_score, places=3,
                             msg= "Expected to see %d cross encoder score but saw %d for document %d"
                             % (expected_score, actual_score, docid))

class TestL2RLambdaMART(unittest.TestCase):

    def test_fit(self):
        ranker = LambdaMART()

        X = [[0, 1],
             [1, 0],
             [0.2, 0.3]]
        y = [2, 0, 1]
        query_group = [3]
        ranker.fit(X, y, query_group)

    def test_predict(self):
        ranker = LambdaMART()

        X_train = [[0, 1],
                   [1, 0],
                   [0.2, 0.3]]
        X_test = [[0, .9],
                  [.9, 0],
                  [0.1, 0.4]]
        y_train = [2, 0, 1]

        # These equal ranks are just because we have toy input data
        y_test = [0, 0, 0]

        query_group = [3]
        ranker.fit(X_train, y_train, query_group)

        ranks = ranker.predict(X_test)
        ranks = list(ranks)
        # print('output:', ranks, ranks.shape)

        self.assertListEqual(ranks, y_test, "Predicted Ranking did not match")


if __name__ == '__main__':
    unittest.main()
