"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """

    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        TODO (HW3): We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.

        """
        # 1. Tokenize query

        # 2. Fetch a list of possible documents from the index

        # 3. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)

        # 4. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]

        raise NotImplementedError


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

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        return NotImplementedError


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
        return NotImplementedError


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        return NotImplementedError


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        return NotImplementedError


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        return NotImplementedError


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    pass


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
        pass
