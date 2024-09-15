'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import json
import os

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    # SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents
        self.term_metadata = {}  # metadata like term count, document frequency of the terms

        self.index = {}  # the index 

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        self.statistics['unique_token_count'] = 0  # the number of unique tokens in the index
        self.statistics['total_token_count'] = 0 # the number of total tokens in the index, i.e., the sum of the lengths of all documents
        self.statistics['stored_total_token_count'] = 0 # the number of total tokens in the index excluding filter tokens
        self.statistics['number_of_documents'] = 0 # the number of documents indexed
        self.statistics['mean_document_length'] = 0 # the mean number of tokens in a document including filter tokens
    
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        doc_length = 0
        stored_doc_length = 0   
        unique_tokens = set()

        # 1. update the self.index
        for token in tokens:
            doc_length += 1
            if token is not None:
                stored_doc_length += 1
                # vocab
                self.statistics['vocab'][token] += 1
                # index
                if token not in self.index:
                    self.index[token] = {}
                    self.index[token].setdefault(docid, 0)
                elif docid not in self.index[token]:
                    self.index[token].setdefault(docid, 0)
                self.index[token][docid] += 1 
                
                unique_tokens.add(token)
                

        # 2. update doc_metadata
        self.document_metadata[docid] = {
            "unique_tokens": len(unique_tokens),
            "length": doc_length,
            "sotred_length": stored_doc_length,
            "unique_tokens_list": list(unique_tokens)
        }
        
        # 3. update the self.statistics (including term metadata)
        self.vocabulary.update(unique_tokens)
        self.statistics['total_token_count'] += doc_length
        self.statistics['stored_total_token_count'] += stored_doc_length
        self.statistics['number_of_documents'] += 1
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        self.statistics['unique_token_count'] = len(self.vocabulary)
        
    def remove_doc(self, docid: int) -> None:
        if docid in self.index:
            # 1. update the self.index
            self.statistics['total_token_count'] -= self.document_metadata[docid]['length']
            self.statistics['stored_total_token_count'] -= self.document_metadata[docid]['sotred_length']
            self.statistics['number_of_documents'] -= 1
            if self.statistics['number_of_documents'] == 0:
                self.statistics['mean_document_length'] = 0
            else:
                self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
            
            # 2. update the self.statistics (including term metadata)
            for token in self.document_metadata[docid]['unique_tokens_list']:
                self.statistics['vocab'][token] -= self.index[token][docid]
                if self.statistics['vocab'][token] == 0:
                    del self.statistics['vocab'][token]
                    self.vocabulary.remove(token)
                    del self.index[token]
                else:
                    del self.index[token][docid]
                
            self.statistics['unique_token_count'] = len(self.vocabulary)
            
            # 3. update doc_metadata
            del self.document_metadata[docid]
        
    def get_postings(self, term: str) -> list:
        if term not in self.vocabulary:
            return []
        return [(docid, self.index[term][docid]) for docid in self.index[term]]
    
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        if doc_id not in self.document_metadata:
            return {
                "unique_tokens": 0,
                "length": 0
            }
        return {
            "unique_tokens": self.document_metadata[doc_id]['unique_tokens'],
            "length": self.document_metadata[doc_id]['length']
        }
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        if term not in self.statistics['vocab']:
            return {
                "term_count": 0,
                "doc_frequency": 0
            }
        return {
                "term_count": self.statistics['vocab'][term],
                "doc_frequency": self.index[term].__len__()
            }
        
    def get_statistics(self) -> dict[str, int]:
        result = {
            "unique_token_count": self.statistics['unique_token_count'],
            "total_token_count": self.statistics['total_token_count'],
            "stored_total_token_count": self.statistics['stored_total_token_count'],
            "number_of_documents": self.statistics['number_of_documents'],
            "mean_document_length": self.statistics['mean_document_length']
        }
        return result
        
    def save(self, dir) -> None:
        # 检查路径是否存在
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_path = os.path.join(dir, self.statistics['index_type'] + '.json')
        
        with open(file_path, 'w') as f:
            dict = {
                    'statistics': self.statistics, 
                    'index': self.index, 
                    'vocabulary': list(self.vocabulary), 
                    'document_metadata': self.document_metadata
            }
            json.dump(dict, f, indent=4)
            
    def load(self, dir) -> None:
        file_path = os.path.join(dir, self.statistics['index_type'] + '.json')
        with open(file_path, 'r') as f:
            dict = json.load(f)
            self.statistics = dict['statistics']
            self.index = dict['index']
            self.vocabulary = set(dict['vocabulary'])
            self.document_metadata = dict['document_metadata']
        

class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        doc_length = 0
        stored_doc_length = 0   
        unique_tokens = set()

        # 1. update the self.index
        for position, token in enumerate(tokens):
            doc_length += 1
            if token is not None:
                stored_doc_length += 1
                # vocab
                self.statistics['vocab'][token] += 1
                # index
                if token not in self.index:
                    self.index[token] = {}
                    self.index[token].setdefault(docid, [])
                elif docid not in self.index[token]:
                    self.index[token].setdefault(docid, [])
                self.index[token][docid].append(position)
                
                unique_tokens.add(token)
                

        # 2. update doc_metadata
        self.document_metadata[docid] = {
            "unique_tokens": len(unique_tokens),
            "length": doc_length,
            "sotred_length": stored_doc_length,
            "unique_tokens_list": list(unique_tokens)
        }
        
        # 3. update the self.statistics (including term metadata)
        self.vocabulary.update(unique_tokens)
        self.statistics['total_token_count'] += doc_length
        self.statistics['stored_total_token_count'] += stored_doc_length
        self.statistics['number_of_documents'] += 1
        if self.statistics['number_of_documents'] == 0:
            self.statistics['mean_document_length'] = 0
        else:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        self.statistics['unique_token_count'] = len(self.vocabulary)
        
    def remove_doc(self, docid: int) -> None:
        if docid in self.index:
            # 1. update the self.index
            self.statistics['total_token_count'] -= self.document_metadata[docid]['length']
            self.statistics['stored_total_token_count'] -= self.document_metadata[docid]['sotred_length']
            self.statistics['number_of_documents'] -= 1
            if self.statistics['number_of_documents'] == 0:
                self.statistics['mean_document_length'] = 0
            else:
                self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
            
            # 2. update the self.statistics (including term metadata)
            for token in self.document_metadata[docid]['unique_tokens_list']:
                self.statistics['vocab'][token] -= len(self.index[token][docid])
                if self.statistics['vocab'][token] == 0:
                    del self.statistics['vocab'][token]
                    self.vocabulary.remove(token)
                    del self.index[token]
                else:
                    del self.index[token][docid]
                
            self.statistics['unique_token_count'] = len(self.vocabulary)
            
            # 3. update doc_metadata
            del self.document_metadata[docid]
        
    def get_postings(self, term: str) -> list:
        if term not in self.vocabulary:
            return []
        return [(docid, len(self.index[term][docid]), self.index[term][docid]) for docid in self.index[term]]
        
    

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                        document_preprocessor: Tokenizer, stopwords: set[str],
                        minimum_word_frequency: int, text_key = "text",
                        max_docs: int = -1, ) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.

        Returns:
            An inverted index
        
        '''
        def filter_2_None(token_list, word_count, min_freq, stop_words):
            new_list = [word if (word not in stop_words and word_count[word] >= min_freq) else None for word in token_list]
            return new_list
        
        iindex = None
        if index_type == IndexType.BasicInvertedIndex:
            iindex = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            iindex = PositionalInvertedIndex()
            
        tokenizer = document_preprocessor 
        
        docid_2_toekns = {}
        word_count = Counter()
        with open(dataset_path, 'r') as f:
            for idx, line in enumerate(f):
                if max_docs != -1 and idx >= max_docs:
                    break
                data = json.loads(line)
                tokens = tokenizer.tokenize(data[text_key])
                docid_2_toekns[data["docid"]] = tokens
                word_count.update(tokens)   
                
            for docid, tokens in docid_2_toekns.items():
                tmp = filter_2_None(tokens, word_count, minimum_word_frequency, stopwords)
                iindex.add_doc(docid, tmp)
                
        return iindex
        
# TODO for each inverted index implementation, use the Indexer to create an index with the first 10, 100, 1000, and 10000 documents in the collection (what was just preprocessed). At each size, record (1) how
# long it took to index that many documents and (2) using the get memory footprint function provided, how much memory the index consumes. Record these sizes and timestamps. Make
# a plot for each, showing the number of documents on the x-axis and either time or memory
# on the y-axis.

'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1
    
    def save(self):
        print('Index saved!')

