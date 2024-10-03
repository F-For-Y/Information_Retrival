'''
Author: Prithvijit Dasgupta
Modified by: Zim Gong
This file is a template code file for the Search Engine. 
'''
# your library imports go here
from collections import defaultdict, Counter
import os
import json
import pickle
import gzip
from tqdm import tqdm

# project library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType
from ranker import *
from models import BaseSearchEngine, SearchResponse
from network_features import NetworkFeatures
from l2r import L2RRanker


DATA_PATH = 'data/'  # TODO: Set this to the path to your data folder
CACHE_PATH = '__cache__/'  # Set this to the path of the cache folder

STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
DATASET_PATH = DATA_PATH + 'wikipedia_200k_dataset.jsonl.gz'
NETWORK_STATS_PATH = DATA_PATH + 'network_stats.csv'
EDGELIST_PATH = DATA_PATH + 'edgelist.csv'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'


class SearchEngine(BaseSearchEngine):
    def __init__(self, max_docs: int = -1, ranker: str = 'BM25', l2r: bool = True) -> None:
        # This is the pipeline of the search engine. Feel free to modify this code.
        # For reference, the pipeline consists of the following steps:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        # 2. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        # 3. Initialize the ranker using the Ranker class and the index
        # 4. Initialize the pipeline with the ranker

        print('Initializing Search Engine...')
        self.stopwords = set()
        with open(STOPWORD_PATH, 'r') as f:
            for line in f:
                self.stopwords.add(line.strip())

        print('Loading indexes...')
        self.preprocessor = RegexTokenizer('\w+')

        self.main_index = Indexer.create_index(
            IndexType.BasicInvertedIndex, DATASET_PATH, self.preprocessor,
            self.stopwords, 50, text_key='text', max_docs=max_docs
        )

        print('Loading ranker...')
        self.set_ranker(ranker)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'BM25') -> None:
        if ranker == 'BM25':
            self.scorer = BM25(self.main_index)
        elif ranker == "WordCountCosineSimilarity":
            self.scorer = WordCountCosineSimilarity(self.main_index)
        elif ranker == "DirichletLM":
            self.scorer = DirichletLM(self.main_index)
        elif ranker == "PivotedNormalization":
            self.scorer = PivotedNormalization(self.main_index)
        elif ranker == "TF_IDF":
            self.scorer = TF_IDF(self.main_index)
        else:
            raise ValueError("Invalid ranker type")
        self.ranker = Ranker(
            self.main_index, self.preprocessor, self.stopwords,
            self.scorer, raw_text_dict=None)

        if self.l2r:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_l2r(self, l2r: bool = True) -> None:
        if self.l2r == l2r:
            return
        if not l2r:
            self.pipeline = self.ranker
        else:
            print('Loading categories...')
            if not os.path.exists(CACHE_PATH + 'docid_to_categories.pkl'):
                docid_to_categories = defaultdict()
                with gzip.open(DATASET_PATH, 'rt') as f:
                    for line in tqdm(f):
                        data = json.loads(line)
                        docid_to_categories[data['docid']] = data['categories']
                pickle.dump(
                    docid_to_categories,
                    open(CACHE_PATH + 'docid_to_categories.pkl', 'wb')
                )
            else:
                docid_to_categories = pickle.load(
                    open(CACHE_PATH + 'docid_to_categories.pkl', 'rb')
                )

            print('Loading recognized categories...')
            category_counts = Counter()
            for categories in tqdm(docid_to_categories.values()):
                category_counts.update(categories)
            self.recognized_categories = set(
                [category for category, count in category_counts.items()
                 if count > 1000]
            )
            if not os.path.exists(CACHE_PATH + 'doc_category_info.pkl'):
                self.doc_category_info = defaultdict()
                for docid, categories in tqdm(docid_to_categories.items()):
                    self.doc_category_info[docid] = [
                        category for category in categories if category in self.recognized_categories
                    ]
                pickle.dump(
                    self.doc_category_info,
                    open(CACHE_PATH + 'doc_category_info.pkl', 'wb')
                )
            else:
                self.doc_category_info = pickle.load(
                    open(CACHE_PATH + 'doc_category_info.pkl', 'rb')
                )
            del docid_to_categories, category_counts

            print('Loading network features...')
            self.network_features = defaultdict()
            if not os.path.exists(NETWORK_STATS_PATH):
                nf = NetworkFeatures()
                graph = nf.load_network(EDGELIST_PATH)
                net_feats_df = nf.get_all_network_statistics(graph)
                del graph
                net_feats_df.to_csv(NETWORK_STATS_PATH, index=False)
                for idx, row in tqdm(net_feats_df.iterrows()):
                    for col in ['pagerank', 'hub_score', 'authority_score']:
                        self.network_features[row['docid']][col] = row[col]
                del net_feats_df
            else:
                with open(NETWORK_STATS_PATH, 'r') as f:
                    for idx, row in tqdm(enumerate(f)):
                        if idx == 0:
                            continue
                        splits = row.strip().split(',')
                        self.network_features[int(splits[0])] = {
                            'pagerank': float(splits[1]),
                            'hub_score': float(splits[2]),
                            'authority_score': float(splits[3])
                        }

            print('Loading L2R ranker...')
            self.pipeline = L2RRanker(
                self.main_index, self.title_index, self.preprocessor,
                self.stopwords, self.ranker, self.fe
            )

            print('Training L2R ranker...')
            self.pipeline.train(RELEVANCE_TRAIN_PATH)
            self.l2r = True

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine()
    return search_obj


if __name__ == '__main__':
    model = initialize()
