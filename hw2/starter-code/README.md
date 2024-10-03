# Homework 2: Learning to Rank

## File Walkthrough
### `document_preprocessor.py`

#### `Tokenizer` class

- _General Comment_: All Tokenizer subclasses should follow the pattern of using the base class's `postprocess` once they have applied their class-specific logic for how to turn the string into an initial list of tokens. You only need to worry about implementing/changing the `RegexTokenzer` for HW2 though!

- `__init__(self, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`
  * Argument `lowercase: bool` (default `True`) to enable lowercasing as option
  * `multiword_expressions: list[str]` (default `None`) supplies the MWE directly into the `Tokenizer`
   
- `find_and_replace_mwes(self, input_tokens: list[str]) -> list[str]`
  * **NOTE:** you do not need to implement this function since homework 2 is not using MWEs!!!

- `postprocess(self, input_tokens: list[str]) -> list[str]`
  * apply MWE and lowercasing, if enabled (**NOTE** Only lowercase is enabled)
  * After tokenizing in each of the child classes, pass through this function

#### `SplitTokenizer` class
- `__init__(self, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`

#### `RegexTokenizer` class
- `__init__(self, token_regex: str, lowercase: bool= True, multiword_expressions: list[str]=None) -> None`
  * `token_regex: str`, the regular expression used

#### 

---

### `indexing.py`

#### `BasicInvertedIndex`

- `__init__(self)`

- `save(self, index_directory_name)`
  * Save should store _all_ state needed to rank documents. If any metadata is needed, it should be stored in some file. We recommend having at least two other files to save the document metadata and the collection statistics.

- `load(self, index_directory_name)`

#### `Indexer`

- `def create_index(index_type: IndexType, dataset_path: str, document_preprocessor: Tokenizer, stopwords: set[str],         minimum_word_frequency: int, text_key="text", max_docs=-1)`
  * `stopwords: set[bool]`，either a set of stopwords is provided or `None` or an empty `set` is provided to not perform stopwords filtering
  * `text_key` 
    - This argument specifies which key to use in the dataset's JSON object for getting text when creating the index. It defaults to "text" but when you create an index for the title, you'll need to change its value
  * `max_docs`
    - This specifies the maximum number of document you should process from the dataset (ignoring the rest). If the argument's value is -1, you use all the documents. This argument is helpful for testing settings where you want an index of real data but you don't want to load in all the data.
  * **Important Note** `minimum_word_frequency` needs to be computed based on frequencies at the collection-level, _not_ at the document level. This means if some minimum is required, the entire collection needs to be read and tokenized first to count word frequencies and _then_ the collection should be re-read to filter based on these counts. 

---

### `ranker.py`

#### `Ranker` class

- `__init__(self, index, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:`
  * `stopwords: set[str]`: supply the stopwords directly into the `Ranker`
    
- `query(self, query: str) -> list[dict]`
  * During the process of retrieving the list of postings for each of the terms in the query, your code should accumulate the word frequencies for each of the documents in the postings (from the data in the list) for use with the `RelevanceScorer.score`  method. This means that you should create a mapping from a document ID to a dictionary of the counts of the query terms in that document. Note that this is _not_ the full count of all words' frequencies for that document---just the query terms. 
  * You will pass each document's term-frequency dictionary to the `RelevanceScorer` as input. Again, note that this is just a _subset_ of the document's terms that has been pre-filtered to just those terms in the query.
 
#### `RelevanceScorer` class
- `score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float`
  * `doc_word_counts`: a dictionary containing the frequencies of all the query words in the document. 
  * The `score` function returns _only_ a `float`: a score for how relevant the document is, where higher scores are more relevant
    * **NOT** `{'docid': docid, 'score': score}`

---

### `pipeline.py` 

#### `SearchEngine`

- As a quick reminder, this code will let you run your search engine locally!

---

### `relevance.py`

- `map_score(search_result_relevances: list[int], cut_off=10) -> float`

- `ndcg_score(search_result_relevances: list[float], ideal_relevance_score_ordering: list[float], cut_off=10)`

- `run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]`

---

## New classes and functions in Assignment 2

Below is a high level summary of the classes and some notes

### `l2r.py`

#### `L2RFeatureExtractor`

- This class is responsible for turning query-document pairs into a vector of features that could indicate how relevant the document is.
- You should work on this class first 
- You can (and should) make use of your existing classes when generating features, e.g., calculating TF-IDF

#### `LambdaMART`

- This class is responsible for wrapping the `LightGBM` model and calling the appropriate training and predicting functions on it. It's not a fancy class and you have to do minimal implementation for it to work---just wrapping functions.
- Once you have the features correctly extracted, you can work on this class.

#### `L2RRanker`

- This class is where you'll support the basic `Ranker`-like functions using a `LambdaMART` learning to rank model under the hood. 
- You should work on this class _after_ you have the `L2RFeatureExtractor` class finished and debugged

### `network_features.py`

#### `NetworkFeatures`

- This class is going to generate the PageRank and HITS features for all the documents using the link network 
- You'll run this class _separately_ from the main part of the IR system to generate the network features _once_. You should not re-generate these every time you start.
- Most of this class is just dealing with the `sknetwork` library (scikit-network) to load a network and call functions
- The hardest part of this class is figuring out how to load in the network, which will require you to read the documentation and think carefully about how to be memory efficient when loading a network of this size. We estimate that the entire network construction memory requirement is under 5GB based on tests with the reference implementation.

### `Interactive_Example.ipynb `

- This is an optional Jupyter notebook that can help you walk through the various steps you need to run everything. Feel free to use this for interactive debugging and development, in addition to the unit tests.

---

## How to use the public test cases

- To run individual test cases, in your terminal, run:
  * `python [filename] [class].[function]`
  * ex: `python test_relevance_scorers.py TestRankingMetrics.test_bm25_single_word_query`
 
- To run one class's tests from file, in your terminal, run:
  * `python [filename] [class] -vvv`
  * ex: `python test_indexing.py TestBasicInvertedIndex -vvv`

- To run all the tests from file, in your terminal, run:
  * `python [filename] -vvv`
  * ex: `python test_indexing.py -vvv`


- To add your own test cases, the basic structure is as follows:
  
```
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
  
```
More information can be found [here](https://docs.python.org/3/library/unittest.html).
