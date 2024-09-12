import unittest
import sys
import os
import bisect
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# modules being tested
from document_preprocessor import RegexTokenizer
from indexing import IndexType, InvertedIndex, BasicInvertedIndex, Indexer, PositionalInvertedIndex

############ =======Test BasicInvertedIndex=========== ############


class TestBasicInvertedIndex(unittest.TestCase):
    'Test Basic Inverted Index'

    def create_new_index(self) -> InvertedIndex:
        iindex = BasicInvertedIndex()
        return iindex 

    def test_create_basic_index(self):
        iindex = BasicInvertedIndex()
        self.assertNotEqual(None, iindex, 'BasicInvertedIndex creation failed')

    def test_add_doc(self):
        '''Test adding document to index'''

        index = self.create_new_index()

        # Test adding a document to the index and retrieving its postings
        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        doc_id = 2
        tokens = ["apple", "banana", "cherry"]
        index.add_doc(doc_id, tokens)

        # Retrieve postings for a term
        term = "apple"
        postings = index.get_postings(term)
        docid = index.get_doc_metadata(doc_id).get('index_doc_id', doc_id)

        self.assertEqual(len(postings), 2)

        # Assertions
        self.assertIsInstance(index.get_term_metadata(term), dict)
        docid_idx = bisect.bisect_left(postings, docid, key=lambda x: x[0])
        self.assertEqual([t[0] for t in postings], sorted([t[0] for t in postings]))
        self.assertEqual(docid, postings[docid_idx][0])

        # Check term frequency in the document
        self.assertEqual(postings[docid_idx][0], 2)
        self.assertEqual(postings[docid_idx][1], 1)

    def test_add_empty_doc(self):
        '''Test adding empty document to index'''
        # Test adding an empty document to the index and retrieving its postings

        index = self.create_new_index()
        doc_id = 1
        tokens = []
        index.add_doc(doc_id, tokens)

        # Retrieve document metadata for a document
        doc_metadata = index.get_doc_metadata(doc_id)

        # Assertions
        self.assertEqual(index.get_statistics()['unique_token_count'], 0)
        self.assertIsNotNone(doc_metadata)

    def test_no_unique_tokens(self):
        '''Test adding document with no unique tokens'''
        # Test with a document containing no unique tokens
        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "apple", "apple"]
        index.add_doc(doc_id, tokens)
        postings = index.get_postings("apple")
        docid = index.get_doc_metadata(doc_id).get('index_doc_id', doc_id)
        docid_idx = bisect.bisect_left(postings, docid, key=lambda x: x[0])

        # Assertions
        self.assertTrue(index.get_statistics()['unique_token_count'], 1)
        # Check term frequency in the document
        self.assertEqual(postings[docid_idx][1], 3)

    def test_nonexistent_tokens(self):
        '''Test adding document to index and retrieving non-existent term'''
        # Test with a document containing no unique tokens

        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "orange", "banana"]
        index.add_doc(doc_id, tokens)
        # Test with a non-existent term
        term_non_existent = "nonexistent"

        # Assertions
        self.assertEqual(index.get_statistics()['unique_token_count'], 3)
        try:
            arr = index.get_postings(term_non_existent)
            if arr == []:
                self.assertEqual(arr, [])
            else:
                self.assertIsNone(arr)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_get_statistics(self):
        '''Test getting statistics with some expected keys'''
        # Test the get_statistics method
        # Add multiple documents to the index

        index = self.create_new_index()

        index.add_doc(1, ["apple", "banana"])
        index.add_doc(2, ["cherry", "banana"])
        index.add_doc(3, ["apple", "cherry", "date"])

        # Calculate expected statistics
        expected_stats = {
            'unique_token_count': 4,  # apple, banana, cherry, date
            'total_token_count': 7,   # Total tokens in all documents
            'number_of_documents': 3,
            'mean_document_length': 7 / 3  # Average document length
        }

        # Retrieve and compare statistics
        stats = index.get_statistics()
        self.assertEqual(stats, stats | expected_stats)

    def test_empty_statistics(self):
        '''Test getting statistics for empty index'''
        # Test statistics with an empty index (no documents)
        index = self.create_new_index()

        empty_stats = index.get_statistics()

        # Assertions
        expected_stats = {
            'unique_token_count': 0,
            'total_token_count': 0,
            'number_of_documents': 0,
            'mean_document_length': 0
        }
        self.assertEqual(empty_stats, empty_stats | expected_stats)

    def test_remove_doc(self):
        '''Test removing document'''

        index = self.create_new_index()

        # Test removing a document from the index
        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)
        doc_id = 2
        tokens = ["orange", "banana", "cherry"]

        # Remove the document
        to_remove = 1
        index.remove_doc(to_remove)

        # Assertions
        try:
            arr = index.get_doc_metadata(doc_id)
            if arr == {}:
                self.assertEqual(arr, {})
            else:
                self.assertIsNone(arr)
        except Exception as e:
            self.assertIsInstance(e, Exception)

        try:
            arr = index.get_postings('apple')
            if arr == []:
                self.assertEqual(arr, [])
            else:
                self.assertIsNone(arr)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_remove_nonexistent_doc(self):
        '''Test removing non-existent document'''

        index = self.create_new_index()

        # Test removing a document from the index
        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        # Remove the document
        to_remove = 2

        # Test removing a non-existent document (should raise a KeyError)
        with self.assertRaises(Exception):
            index.remove_doc(
                to_remove, 'This is a risky \'delete\' operation and must raise an exception.')

    def test_term_metadata(self):
        '''Test getting term metadata'''

        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        # Fetch term
        term_metadata = index.get_term_metadata('apple')

        # Assertion
        for key, value in term_metadata.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, int)

    def test_nonexistent_term_metadata(self):
        '''Test getting nonexistent term metadata'''

        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        # Assertion
        try:
            d = index.get_term_metadata('orange')
            if d == {}:
                self.assertEqual(d, {})
            else:
                self.assertIsNone(d)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_doc_metadata(self):
        '''Test getting doc metadata'''

        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        # Fetch doc
        doc_metadata = index.get_doc_metadata(doc_id)

        # Assertion
        for key, value in doc_metadata.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, int)

    def test_nonexistent_doc_metadata(self):
        '''Test getting nonexistent doc metadata'''

        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)

        # Assertion
        try:
            d = index.get_doc_metadata(2)
            if d == {}:
                self.assertEqual(d, {})
            else:
                self.assertIsNone(d)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_index_save(self):
        '''Test saving index'''

        index_dir  = 'tmp-basic-index-dir'
        index = self.create_new_index()

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)
        index.save(index_dir)

        self.assertGreater(len(os.listdir(index_dir)), 0)

    def test_index_load(self):
        '''Test loading index'''

        index_dir  = 'tmp-basic-index-dir'
        index = self.create_new_index()
        # index.save(index_dir)

        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        index.add_doc(doc_id, tokens)
        index.save(index_dir)

        test_new_index = self.create_new_index()
        test_new_index.load(index_dir)

        term_metadata = test_new_index.get_term_metadata('apple')
        doc_metadata = test_new_index.get_doc_metadata(doc_id)
        postings = test_new_index.get_postings('apple')

        self.assertGreater(len(os.listdir(index_dir)), 0)
        self.assertIsNotNone(term_metadata)
        self.assertIsNotNone(doc_metadata)
        self.assertIsNotNone(postings)

############ =======Test Indexer for BasicInvertedIndex =========== ############

# Create a mock document preprocessor that tokenizes without filtering

class TestIndexer_BasicInvertedIndex(unittest.TestCase):
    
    def setUp(self) -> None:
        self.index_name = 'test_index'
        self.tokenizer = RegexTokenizer('\w+')
        self.index_type = IndexType.InvertedIndex

    def test_index_vocabularly(self):

        index = Indexer.create_index(self.index_type, 'dataset_2.jsonl', self.tokenizer, set(), 0)
        stats = index.get_statistics()
        
        expected_stats = {'unique_token_count': 10,
                          'total_token_count': 28,
                          'number_of_documents': 3,
                          'mean_document_length': 28/3}
        
        for k, v in expected_stats.items():
            self.assertEqual(stats[k], expected_stats[k], 
                '%s was not what was expected %d' % (k, expected_stats[k]))

    def test_create_index_no_stopwords_no_mwf(self):
        '''Test creating an index with no stopwords and no minimum word frequency'''

        index = Indexer.create_index(self.index_type, 'dataset_2.jsonl', self.tokenizer, set(), 0)
        stats = index.get_statistics()
        expected_stats = {'unique_token_count': 10,
                          'total_token_count': 28,
                          'number_of_documents': 3,
                          'mean_document_length': 28/3}
        
        for k, v in expected_stats.items():
            self.assertEqual(stats[k], expected_stats[k], 
                '%s was not what was expected %d' % (k, expected_stats[k]))

    def test_create_index_no_stopwords_with_mwf(self):
        '''Test creating an index with no stopwords and minimum word frequency = 2'''
        index = Indexer.create_index(self.index_type, 'dataset_2.jsonl', self.tokenizer, set(), 2)
        stats = index.get_statistics()
        expected_stats = {'unique_token_count': 5,
                          'total_token_count': 28,
                          'number_of_documents': 3,
                          'mean_document_length': 28/3}
        
        for k, v in expected_stats.items():
            self.assertEqual(stats[k], expected_stats[k], 
                '%s was not what was expected %d' % (k, expected_stats[k]))

    def test_create_index_with_stopwords_no_mwf(self):
        '''Test creating an index with stopwords and no minimum word frequency'''

        stopwords = set(['and', 'the'])
        index = Indexer.create_index(self.index_type, 'dataset_2.jsonl', self.tokenizer, stopwords, 0)
        stats = index.get_statistics()
        expected_stats = {'unique_token_count': 8,
                          'total_token_count': 28,
                          'number_of_documents': 3,
                          'mean_document_length': 28/3}
        for k, v in expected_stats.items():
            self.assertEqual(stats[k], expected_stats[k], 
                '%s was not what was expected %d' % (k, expected_stats[k]))


############ ======= Test PositionalInvertedIndex =========== ############
class TestPositionalInvertedIndex(TestBasicInvertedIndex):
    'Test Postional Inverted Index'

    def setUp(self):
        # Create a temporary directory for testing and initialize an index
        self.index = PositionalInvertedIndex()

    def test_get_postings(self):
        '''Test if positional inverted index has positions'''
        doc_id = 1
        tokens = ["apple", "banana", "apple", "cherry"]
        self.index.add_doc(doc_id, tokens)

        doc_id = 2
        tokens = ["apple", "banana", "cherry"]
        self.index.add_doc(doc_id, tokens)

        # Retrieve postings for a term
        term = "apple"
        postings = self.index.get_postings(term)
        docid = self.index.get_doc_metadata(doc_id).get('index_doc_id', doc_id)

        # Assertions
        self.assertIsInstance(self.index.get_term_metadata(term), dict)
        docid_idx = bisect.bisect_left(postings, docid, key=lambda x: x[0])
        self.assertEqual(docid, postings[docid_idx][0])
        # Check term frequency in the document
        self.assertEqual(postings[docid_idx][1], 1)
        self.assertEqual(postings[docid_idx][2], [0])


if __name__ == '__main__':
    unittest.main()