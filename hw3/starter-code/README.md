## General strategy for Assignment 3

1. Implement the doc2query part (in `document_preprocessor.py`)

2. Implement the `VectorRanker` (in `vector_ranker.py`)

3. Do the bi-encoder experiments from the paper

4. Implement the `CrossEncoderScorer` in `ranker.py`

5. Integrate all of these into the IR system.

6. Perform the full IR system experiments

## Changes from Assignment 2

### Data Changes

- We will continue using the 'wikipedia_200k_dataset.jsonl' for HW3 as our base corpus.

- We will continue using the relevance dataset from the previous homework.

- (**NEW**) We have added a new file called `doc2query.csv` which has the queries generated for all of the 200k documents using `doc2query/msmarco-t5-base-v1`. *This should be used while indexing and not in document preprocessor.*

- (**NEW**) We have added a new file called `wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy` which has all the document embeddings meant to be used in your `VectorRanker`. Use `numpy.load` to load the numpy matrix and the embedding insertion order follows the document order in the `wikipedia_200k_dataset.jsonl`.

- For convenience, we've added the document ids for the 200K articles in the same order as in the JSON in `document-ids.txt`. You can use this in your experiments, if needed (e.g., the bi-encoder tests).

---

### `document_preprocessor.py`

There are no changes to the `Tokenizer` classes. `Doc2QueryAugmenter` has been added.

#### `Doc2QueryAugmenter` class

- _General Comment_: This class should be a functional piece of code which takes in a doc2query model and can generate queries from a piece of text. **DO NOT** waste your time generating queries for all 200k documents. It would take days to finish on your laptop. Rather, this is to check your skills with HuggingFace transformers and pre-trained models. For downstream tasks such as index augmentation with the queries, use `doc2query.csv`.

**Check the code comments for more clarity on what to do**

---

### `indexing.py`

There are no changes to `InvertedIndex` classes. `Indexer` has an additive change.

#### `Indexer`

- `def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None)`
  * This function now has an *optional* argument `doc_augment_dict`. This dict should be created from the `doc2query.csv` The keys are the document id and the values are the list of queries for a particular document. The augmentation of the document should happen before all of the preprocessing steps, i.e., before stopwords removal or minimum word filtering. Note that index-builders do not need to augment their documents so your code should support the cases where `doc_augment_dict` is or isn't provided.

---

### `ranker.py`

There are no changes to `RelevanceScorer` and `Ranker` classes. 

#### `CrossEncoderScorer` class

- _General Comment_: This class uses a pre-trained cross-encoder model from the Sentence Transformers package to score a given query-document pair. Since our cross-encoder model can process a maximum of 512 tokens, you need to create a dictionary that maps the document ID to a text (string) with the first 500 words in the document as instructed in Section 5 of the specification before working on this part. Then, you should pass the dictionary to the class as an argument. Note that the cross-encoder model receives raw strings as input; you should neither filter stopwords nor tokenize a query or document text before feeding them into the model.

---

### `l2r.py`

There are no major changes in `l2r.py`. However, one small change is that we won't be using `RelevanceScorer` in the L2R initialization. This has been removed in favor of using a `Ranker` object. This `Ranker` can be your traditional `Ranker` or the new `VectorRanker` or even another `L2RRanker`. 

What the `RelevanceScorer` used to do was essentially find and initial ranking of the documents and then, L2RRanker reranked those documents using `LambdaMART`.

But this can be done more easily with `Ranker` objects as they are supposed to find lists of relevant documents

- ` def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor')`
  * the `scorer` has been replaced with `ranker`

## New code files in Assignment 3

### `vector_ranker.py`

#### `vector_ranker.py` implements a vector-based ranker that utilizes pre-trained models from the HuggingFace Transformers library to generate document embeddings and query embeddings.

`VectorRanker` Class inherits from the Ranker class and contains the following:
- `def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray, row_to_docid: list[int]) -> None`:
    - Instantiates the Sentence Transformer model and accepts the following parameters:
      - bi_encoder_model_name (str): The name of a HuggingFace model used to initialize a Sentence Transformer model.
      - encoded_docs (ndarray): A matrix where each row represents an already-encoded document using the same encoding as specified by `bi_encoder_model_name`.
      - row_to_docid (list[int]): A list that maps row numbers to the document IDs corresponding to the embeddings.

 -  `def query(self, query: str) -> list[tuple[int, float]]`:
    - Takes a query string as input, encodes the query into a vector, and calculates the relevance scores between the query and all documents.
    - Returns a sorted list of tuples, where each tuple contains a document ID and its relevance score, with the most relevant documents ranked first.
    - You will be computing the dot product between the query vector and document vectors.
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