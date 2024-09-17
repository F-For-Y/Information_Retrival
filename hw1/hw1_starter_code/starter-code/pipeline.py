'''
Author: Prithvijit Dasgupta
Modified by: Zim Gong
This file is a template code file for the Search Engine. 
'''
# your library imports go here

# project library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType
from ranker import *
from models import BaseSearchEngine, SearchResponse


DATA_PATH = 'data/' # TODO: Set this to the path to your data folder
CACHE_PATH = '__cache__/' # Set this to the path of the cache folder

STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
DATASET_PATH = DATA_PATH + 'wikipedia_200k_dataset.jsonl'


class SearchEngine(BaseSearchEngine):
    def __init__(self, max_docs: int = -1, ranker: str = 'BM25') -> None:
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
            self.stopwords, 50, text_key='text', max_docs=max_docs, load_file_dir='./wiki_index_dir'
        )
        print('Indexing creation complete!')
        
        # self.main_index.save('wiki_index_dir')
        
        # print('Indexing saved to ./wiki_index_dir/')

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
            self.scorer, raw_text_dict = None)
        
        self.pipeline = self.ranker

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
