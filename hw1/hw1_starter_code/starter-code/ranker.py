"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
import math

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict
        self.query_len = 0

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        q_tokens = self.tokenize(query)
        self.query_len = len(q_tokens)
        q_tokens = [token if token not in self.stopwords else "$$" for token in q_tokens]
        q_count = {}
        for token in q_tokens:
            if token in q_count:
                q_count[token] += 1
            else:
                q_count[token] = 1
                
        # print('User query:', q_tokens)

        # 2. Fetch a list of possible documents from the index
        # 用这样的方式的话，仅仅包含stopwords overlap的docid就不会被召回了
        docid_list = []
        for tokens in q_tokens:
            if tokens in self.index.index:
                for docid in self.index.index[tokens]:
                    docid_list.append(docid)
        docid_list = list(set(docid_list))
        
        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        res_list = []
        for docid in docid_list:
            relevance = self.scorer.score(docid, {}, q_count)
            res_list.append((docid, relevance))
        
        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        res_list = sorted(res_list, key=lambda x: x[1], reverse=True)
        # print(res_list[:10])
        return res_list


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        dot_product = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                dot_product += q_count * self.index.index[term].get(docid)
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                dot_product += q_count * len(self.index.index[term].get(docid))

        # 2. Return the score
        return dot_product

# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        # 2. Compute additional terms to use in algorithm
        # 3. For all query_parts, compute score
        # 4. Return the score
        score = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                q_c = query_word_counts[term]
                second_part = math.log(1 + self.index.index[term][docid] / (self.parameters['mu'] * self.index.statistics['vocab'][term] / self.index.statistics['total_token_count']))
                score += q_c * second_part
                
            score += sum(query_word_counts.values()) * math.log(self.parameters['mu'] / (self.parameters['mu'] + self.index.document_metadata[docid]['length']))
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                q_c = query_word_counts[term]
                second_part = math.log(1 + len(self.index.index[term].get(docid)) / (self.parameters['mu'] * self.index.statistics['vocab'][term] / self.index.statistics['total_token_count']))
                score += q_c * second_part
                
            score += sum(query_word_counts.values()) * math.log(self.parameters['mu'] / (self.parameters['mu'] + self.index.document_metadata[docid]['length']))
        
        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
    
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        # 3. For all query parts, compute the TF and IDF to get a score    
        score = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                var_idf = math.log((self.index.statistics['number_of_documents'] - len(self.index.index[term]) + 0.5) / (len(self.index.index[term]) + 0.5))
                var_tf = (self.k1 + 1) * self.index.index[term][docid] / (self.k1 * ((1 - self.b) + self.b * self.index.document_metadata[docid]['length'] / self.index.statistics['mean_document_length']) + self.index.index[term][docid])  
                norm_qtf = (self.k3 + 1) * q_count / (self.k3 + q_count)
                score += var_idf * var_tf * norm_qtf
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                var_idf = math.log((self.index.statistics['number_of_documents'] - len(self.index.index[term]) + 0.5) / (len(self.index.index[term]) + 0.5))
                var_tf = (self.k1 + 1) * len(self.index.index[term][docid]) / (self.k1 * ((1 - self.b) + self.b * self.index.document_metadata[docid]['length'] / self.index.statistics['mean_document_length']) + len(self.index.index[term][docid]))  
                norm_qtf = (self.k3 + 1) * q_count / (self.k3 + q_count)
                score += var_idf * var_tf * norm_qtf

        # 4. Return score
        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        # 2. Compute additional terms to use in algorithm
        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
        score = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                # 4. Return the score
                q_tf = query_word_counts[term]
                norm_tf = (1 + math.log(1 + math.log(self.index.index[term][docid]))) / (1 - self.b + self.b * self.index.document_metadata[docid]['length'] / self.index.statistics['mean_document_length'])
                idf = math.log((self.index.statistics['number_of_documents']+1) / len(self.index.index[term]))
                score += q_tf * norm_tf * idf
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                # 4. Return the score
                q_tf = query_word_counts[term]
                norm_tf = (1 + math.log(1 + math.log(len(self.index.index[term][docid])))) / (1 - self.b + self.b * self.index.document_metadata[docid]['length'] / self.index.statistics['mean_document_length'])
                idf = math.log((self.index.statistics['number_of_documents']+1) / len(self.index.index[term]))
                score += q_tf * norm_tf * idf
        
        # 4. Return the score
        return score


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        # 2. Compute additional terms to use in algorithm
        # 3. For all query parts, compute the TF and IDF to get a score
        score = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                # 4. Return the score
                tf = math.log(1 + self.index.index[term][docid])
                idf = 1 + math.log(self.index.statistics['number_of_documents'] / len(self.index.index[term]))
                score += tf * idf
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                # 4. Return the score
                tf = math.log(1 + len(self.index.index[term][docid]))
                idf = 1 + math.log(self.index.statistics['number_of_documents'] / len(self.index.index[term]))
                score += tf * idf

        # 4. Return the score
        return score


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        
    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0
        q_len = 0
        overlap = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                q_len += q_count
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                # 4. Return the score
                overlap += q_count
                tf = self.index.index[term][docid] / self.index.document_metadata[docid]['length']
                idf = 1 + math.log(self.index.statistics['number_of_documents'] / len(self.index.index[term]))
                score += tf * idf
        else:
            for term, q_count in query_word_counts.items():
                q_len += q_count
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                # 4. Return the score
                overlap += q_count
                tf = len(self.index.index[term][docid]) / self.index.document_metadata[docid]['length']
                idf = 1 + math.log(self.index.statistics['number_of_documents'] / len(self.index.index[term]))
                score += tf * idf
            
        return score * overlap / q_len 
