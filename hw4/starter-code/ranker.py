"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import math
from tqdm import tqdm


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
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

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2,user_id=None) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        # TODO (HW4): If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.

        # TODO (HW4): Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE (HW4): Since you're using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers) which is ok and totally expected.

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

        # 2. Fetch a list of possible documents from the index
        docid_list = []
        for tokens in q_tokens:
            if tokens in self.index.index:
                for docid in self.index.index[tokens]:
                    docid_list.append(docid)
        docid_list = list(set(docid_list))

        # 3. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        res_list = []
        for docid in docid_list:
            relevance = self.scorer.score(docid, {}, q_count)
            res_list.append((docid, relevance))

        # 4. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]
        res_list = sorted(res_list, key=lambda x: x[1], reverse=True)
        
        # 5. check if pseudo-feedback is requested
        pseudo_docs_word_count = {}
        if pseudofeedback_num_docs > 0:
            pseudo_rel_docids = [docid for docid, _ in res_list[:pseudofeedback_num_docs]]
            for docid in pseudo_rel_docids:
                raw_text = self.raw_text_dict[docid]
                tokens = self.tokenize(raw_text)
                # print('raw_text length:', len(tokens))
                tokens = [token for token in tokens if token in self.index.index] # tokens in self.index.index satisfy the filtering condition
                # print('filtered length:', len(tokens))
                for token in tokens:
                    if token in pseudo_docs_word_count:
                        pseudo_docs_word_count[token] += 1
                    else:
                        pseudo_docs_word_count[token] = 1
                        
            # scale according to beta:
            for token in q_count:
                q_count[token] = q_count[token] * pseudofeedback_alpha
            
            for token in pseudo_docs_word_count:
                if token in q_count:
                    q_count[token] += pseudofeedback_beta * pseudo_docs_word_count[token] / len(pseudo_rel_docids)
                else:
                    q_count[token] = pseudofeedback_beta * pseudo_docs_word_count[token] / len(pseudo_rel_docids)
                    
            print("complete pseudo-feedback, start quering")
                    
            # do the query again
            docid_list = []
            for tokens in list(q_count.keys()):
                if tokens != '$$':
                    for docid in self.index.index[tokens]:
                        docid_list.append(docid)
            docid_list = list(set(docid_list))
            
            res_list = []
            for docid in tqdm(docid_list):
                relevance = self.scorer.score(docid, {}, q_count)
                res_list.append((docid, relevance))
                
            res_list = sorted(res_list, key=lambda x: x[1], reverse=True)
            return res_list
                
        else:
            return res_list


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index: InvertedIndex, parameters) -> None:
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


# TODO: Implement unnormalized cosine similarity on word count vectors
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


# TODO (HW4): Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO (HW4): Implement Personalized BM25
        score = 0
        if self.index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and self.index.index[term].get(docid, 0) == 0):
                    continue
                ri = len(self.relevant_doc_index.index.get(term, []))
                R = self.relevant_doc_index.statistics['number_of_documents']
                var_idf_numerator = (ri + 0.5) * (self.index.statistics['number_of_documents'] - len(self.index.index[term]) - R + ri + 0.5)
                var_idf_denominator = (len(self.index.index[term]) - ri + 0.5) * (R - ri + 0.5)
                var_idf = math.log(var_idf_numerator / var_idf_denominator)
                var_tf = (self.k1 + 1) * self.index.index[term][docid] / (self.k1 * ((1 - self.b) + self.b * self.index.document_metadata[docid]['length'] / self.index.statistics['mean_document_length']) + self.index.index[term][docid])  
                norm_qtf = (self.k3 + 1) * q_count / (self.k3 + q_count)
                score += var_idf * var_tf * norm_qtf
        else:
            for term, q_count in query_word_counts.items():
                if term not in self.index.index or (term in self.index.index and len(self.index.index[term].get(docid, [])) == 0):
                    continue
                ri = len(self.relevant_doc_index.index.get(term, []))
                R = self.relevant_doc_index.statistics['number_of_documents']
                var_idf_numerator = (ri + 0.5) * (self.index.statistics['number_of_documents'] - len(self.index.index[term]) - R + ri + 0.5)
                var_idf_denominator = (len(self.index.index[term]) - ri + 0.5) * (R - ri + 0.5)
                var_idf = math.log(var_idf_numerator / var_idf_denominator)
                
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


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# NOTE: This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = raw_text_dict
        self.model = CrossEncoder(cross_encoder_model_name, max_length=512)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)

        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        docs = self.raw_text_dict.get(docid, '')
        if not docs or not query:
            return 0
        pairs = [(query, docs)]
        scores = self.model.predict(pairs)
        return scores[0]
