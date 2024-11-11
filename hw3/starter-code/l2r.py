import lightgbm

from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
import numpy as np
import pandas as pd 
from collections import Counter
from ranker import *
import json, math
from tqdm import tqdm


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker, # 在粗排中这里的ranker可以是BM25, 也可以是VectorRanker
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.scorer = ranker
        self.feature_extractor = feature_extractor

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()
        

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features

        # TODO: Accumulate the token counts for each document's title and content here

        # TODO: For each of the documents, generate its features, then append
        # the features and relevance score to the lists to be returned

        # TODO: Make sure to keep track of how many scores we have for this query in qrels

        for query, doc_rel_list in tqdm(query_to_document_relevance_scores.items()):
            query_parts = self.document_preprocessor.tokenize(query)
            if self.stopwords:
                query_parts = [token for token in query_parts if token not in self.stopwords]

            # Accumulate the token counts for each document's title and content
            doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
            title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)

            num_docs = 0
            for docid, relevance_score in doc_rel_list:
                num_docs += 1
                # Get word counts and generate features
                doc_counts = doc_word_counts.get(docid, {})
                title_counts = title_word_counts.get(docid, {})
                features = self.feature_extractor.generate_features(docid, doc_counts, title_counts, query_parts, raw_query=query)
                if not features:
                    num_docs -= 1
                    continue

                X.append(features)
                y.append(relevance_score)

            # Append the number of documents for this query's qgroups
            qgroups.append(num_docs)

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_counts = {}
        for term in query_parts:
            postings = index.get_postings(term)
            if postings:
                for docid, freq in postings:
                    if docid not in doc_term_counts:
                        doc_term_counts[docid] = {}
                    doc_term_counts[docid][term] = freq
        return doc_term_counts

    def train(self, training_data_filename: str, dev_data_filename: str = None, train_dev_file_dir: str = None) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
            
        Training and Validation Process:
            - 先用BM25 retrieve top 100 documents, 然后让model learn to re-rank
        """
        # TODO: Convert the relevance data into the right format for training data preparation

        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures

        # TODO: Train the model
        # Check if the train file has been preprocessed: train_file_dir
        if train_dev_file_dir is not None: # 为了节约时间，不然重新加载一次又要半小时
            data = np.load(train_dev_file_dir)
            X, y, qgroups = data['X_train'].tolist(), data['Y_label'].tolist(), data['qgroups'].tolist()
            X_dev, y_dev, qgroups_dev = data['X_dev'].tolist(), data['Y_dev_label'].tolist(), data['qgroups_dev'].tolist()
            print(f"Train and Dev Data has been loaded from {train_dev_file_dir}......X_shape: {np.array(X).shape}, y_shape: {np.array(y).shape}, qgroups_shape: {np.array(qgroups).shape}")
            self.model.fit(X, y, qgroups, X_dev, y_dev, np.array(qgroups_dev))
            return 
            
        query_to_document_relevance_scores = {}
        df = pd.read_csv(training_data_filename, encoding='ISO-8859-1')     
        val_dict = {} # {QUERYU: {DOCID: REL}} 这里是train, dev, test set中有标注得 (docid, rel) pair
        for i in range(len(df)):
            val_dict.setdefault(df['query'][i], {})
            val_dict[df['query'][i]][df['docid'][i]] = math.ceil(df['rel'][i])
            
        # we only want top100 documents
        for query in tqdm(val_dict.keys()):
            roughtly_ranked_doc = self.scorer.query(query)[:100]
            for docid, _ in roughtly_ranked_doc:
                if query not in query_to_document_relevance_scores:
                        query_to_document_relevance_scores[query] = []
                if docid in val_dict[query]:
                    query_to_document_relevance_scores[query].append((docid, val_dict[query][docid]))
                else:
                    query_to_document_relevance_scores[query].append((docid, 1)) # 不存在于标注数据中的doc, 默认rel=1
                    
        # for i, row in df.iterrows():
        #     query, docid, relevance = row['query'], row['docid'], row['rel']
        #     if query not in query_to_document_relevance_scores:
        #         query_to_document_relevance_scores[query] = []
        #     query_to_document_relevance_scores[query].append((docid, math.ceil(relevance))) # 这里的rel可能不是整数

        # Prepare the training data
        # print(len(list(query_to_document_relevance_scores.keys())))
        X, y, qgroups = self.prepare_training_data(query_to_document_relevance_scores)
        print(f"Train Data has been prepared......X_shape: {np.array(X).shape}, y_shape: {np.array(y).shape}, qgroups_shape: {np.array(qgroups).shape}") 
        
        # 是否指定dev_data_filename: 
        if dev_data_filename:
            query_to_document_relevance_scores_dev = {}
            df_dev = pd.read_csv(dev_data_filename, encoding='ISO-8859-1')
            val_dict_dev = {} # {QUERYU: {DOCID: REL}} 这里是train, dev, test set中有标注得 (docid, rel) pair
            for i in range(len(df_dev)):
                val_dict_dev.setdefault(df_dev['query'][i], {})
                val_dict_dev[df_dev['query'][i]][df_dev['docid'][i]] = math.ceil(df_dev['rel'][i])
                
            # we only want top100 documents
            for query in tqdm(val_dict_dev.keys()):
                roughtly_ranked_doc = self.scorer.query(query)[:100]
                for docid, _ in roughtly_ranked_doc:
                    if query not in query_to_document_relevance_scores_dev:
                            query_to_document_relevance_scores_dev[query] = []
                    if docid in val_dict_dev[query]:
                        query_to_document_relevance_scores_dev[query].append((docid, val_dict_dev[query][docid]))
                    else:
                        query_to_document_relevance_scores_dev[query].append((docid, 1))

            X_dev, y_dev, qgroups_dev = self.prepare_training_data(query_to_document_relevance_scores_dev)
            if train_dev_file_dir is None:
                np.savez('./cache/rel_train_dev.npz', X_dev = np.array(X_dev), Y_dev_label = np.array(y_dev), qgroups_dev = np.array(qgroups_dev), X_train = np.array(X), Y_label = np.array(y), qgroups = np.array(qgroups))
                print('Dev and Train Data has been saved to ./cache/rel_dev.npy......')
            print(f"Dev Data has been prepared......X_dev_shape: {np.array(X_dev).shape}, y_dev_shape: {np.array(y_dev).shape}, qgroups_dev_shape: {np.array(qgroups_dev).shape}")
            print('Start to train the model with dev data......')
            self.model.fit(X, y, qgroups, X_dev, y_dev, np.array(qgroups_dev))
        else:
            print('Start to train the model without dev data......')
            self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.model.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        if not query: # for empty query
            return []
        query_parts = self.document_preprocessor.tokenize(query)
        if self.stopwords:
            query_parts = [token for token in query_parts if token not in self.stopwords]
        if not query_parts:
            return []
        query_word_counts = dict(Counter(query_parts))

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        # pass these doc-term-counts to functions later, so we need the accumulated representations
        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_parts)
        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_parts)
        if not doc_word_counts:
            return []

        # TODO: Accumulate the documents word frequencies for the title and the main body
        candidate_docs = list(doc_word_counts.keys())

        # TODO: Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model
        candidate_docs_scores = []
        # for docid in candidate_docs:
        #     doc_counts = doc_word_counts.get(docid, {})
        #     score = self.scorer.score(docid, doc_counts, query_word_counts)
        #     candidate_docs_scores.append((docid, score))
        candidate_docs_scores = self.scorer.query(query)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        # candidate_docs_scores.sort(key=lambda x: x[1], reverse=True)
        top_100_docs_scores = candidate_docs_scores[:100] # for non-exist terms

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        X = []
        docids = []
        for docid, _ in top_100_docs_scores:
            doc_counts = doc_word_counts.get(docid, {})
            title_counts = title_word_counts.get(docid, {})
            features = self.feature_extractor.generate_features(docid, doc_counts, title_counts, query_parts, raw_query=query)
            X.append(features)
            docids.append(docid)

        # TODO: Use your L2R model to rank these top 100 documents
        scores = self.model.predict(X)
        reranked_docs = list(zip(docids, scores))

        # TODO: Sort posting_lists based on scores
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        other_docs = candidate_docs_scores[100:]

        # TODO: Return the ranked documents
        return reranked_docs + other_docs


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
        """
        # TODO: Set the initial state using the arguments

        # TODO: For the recognized categories (i.e,. those that are going to be features), considering
        # how you want to store them here for faster featurizing

        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring.
        self.document_index = document_index    
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = {category: idx for idx, category in enumerate(sorted(recognized_categories))}
        self.id_2_recognized_categories = {idx: category for idx, category in enumerate(sorted(recognized_categories))}
        self.docid_to_network_features = docid_to_network_features
        
        self.bm25 = BM25(document_index)
        self.pivot_norm = PivotedNormalization(document_index)
        self.cross_encoder = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.document_metadata[docid].get("sotred_length", 0)

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.document_metadata[docid].get("sotred_length", 0)

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        query_word_counts = Counter(query_parts)
        score = 0
        if index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in index.index or (term in index.index and index.index[term].get(docid, 0) == 0):
                    continue
                tf = math.log(1 + index.index[term][docid])
                score += tf
        else:
            for term, q_count in query_word_counts.items():
                if term not in index.index or (term in index.index and len(index.index[term].get(docid, [])) == 0):
                    continue
                tf = math.log(1 + len(index.index[term][docid]))
                score += tf
                
        return score


    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_counts = Counter(query_parts)
        score = 0
        if index.statistics['index_type'] == 'BasicInvertedIndex':
            for term, q_count in query_word_counts.items():
                if term not in index.index or (term in index.index and index.index[term].get(docid, 0) == 0):
                    continue
                # 4. Return the score
                tf = math.log(1 + index.index[term][docid])
                idf = 1 + math.log(index.statistics['number_of_documents'] / len(index.index[term]))
                score += tf * idf
        else:
            for term, q_count in query_word_counts.items():
                if term not in index.index or (term in index.index and len(index.index[term].get(docid, [])) == 0):
                    continue
                # 4. Return the score
                tf = math.log(1 + len(index.index[term][docid]))
                idf = 1 + math.log(index.statistics['number_of_documents'] / len(index.index[term]))
                score += tf * idf
                
        return score

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        query_word_counts = Counter(query_parts)
        return self.bm25.score(docid, doc_word_counts, query_word_counts)

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        query_word_counts = Counter(query_parts)
        return self.pivot_norm.score(docid, doc_word_counts, query_word_counts)

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has.
        """
        cat_vec = [0] * len(self.recognized_categories)
        for cat in self.doc_category_info[docid]:
            if cat in self.recognized_categories:
                cat_vec[self.recognized_categories[cat]] = 1
            
        return cat_vec
        

    # TODO Pagerank score
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        return self.docid_to_network_features[docid].get("pagerank", 0)

    # TODO HITS Hub score
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        if docid not in self.docid_to_network_features:
            return 0
        return self.docid_to_network_features[docid].get("hub_score", 0)

    # TODO HITS Authority score
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        if docid not in self.docid_to_network_features:
            return 0
        return self.docid_to_network_features[docid].get("authority_score", 0)
    
    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        return self.cross_encoder.score(docid, query)

    # TODO 11: Add at least one new feature to be used with your L2R model.

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str], raw_query : str = None) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """ 
        feature_vector = []

        # TODO: Document Length
        doc_length = self.get_article_length(docid)
        feature_vector.append(doc_length)

        # TODO: Title Length
        title_length = self.get_title_length(docid)
        feature_vector.append(title_length)

        # TODO Query Length
        query_length = len(query_parts)
        feature_vector.append(query_length)

        # TODO: TF (document)
        tf_doc = self.get_tf(self.document_index, docid, doc_word_counts, query_parts)
        feature_vector.append(tf_doc)
        
        # TODO: TF-IDF (document)
        tf_idf_doc = self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts)
        feature_vector.append(tf_idf_doc)

        # TODO: TF (title)
        tf_title = self.get_tf(self.title_index, docid, title_word_counts, query_parts)
        feature_vector.append(tf_title)

        # TODO: TF-IDF (title)
        tf_idf_title = self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts)
        feature_vector.append(tf_idf_title)

        # TODO: BM25
        bm25_score = self.get_BM25_score(docid, doc_word_counts, query_parts)
        feature_vector.append(bm25_score)

        # TODO: Pivoted Normalization
        pivot_score = self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)
        feature_vector.append(pivot_score)

        # TODO: Pagerank
        pagerank_score = self.get_pagerank_score(docid)
        feature_vector.append(pagerank_score)

        # TODO: HITS Hub
        hits_hub_score = self.get_hits_hub_score(docid)
        feature_vector.append(hits_hub_score)

        # TODO: HITS Authority
        hits_authority_score = self.get_hits_authority_score(docid)
        feature_vector.append(hits_authority_score)
        
        # TODO: (HW3) Cross-Encoder Score
        cross_encoder_score = self.get_cross_encoder_score(docid, raw_query)
        feature_vector.append(cross_encoder_score)

        # TODO: Add at least one new feature to be used with your L2R model.
        # Additional Feature: Query Term Coverage in Document
        term_coverage = len([term for term in query_parts if (term in self.document_index.index and self.document_index.index[term].get(str(docid), 0) != 0)]) / len(query_parts)
        feature_vector.append(term_coverage)

        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.
        category_features = self.get_document_categories(docid)
        feature_vector.extend(category_features)

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": -1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self,  X_train, y_train, qgroups_train, X_val=None, y_val=None, qgroups_val=None):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        if X_val:
            eval_set = [(np.asarray(X_val), np.asarray(y_val))]
            self.model.fit(
                np.asarray(X_train), np.asarray(y_train), group=np.asarray(qgroups_train),
                eval_set=eval_set, eval_group=[np.asarray(qgroups_val)],
                eval_metric="ndcg", eval_at=[5, 10, 15]
            )
        else:
            self.model.fit(np.asarray(X_train), np.asarray(y_train), group=np.asarray(qgroups_train))
        pass

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)