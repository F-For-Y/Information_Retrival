## General strategy for Assignment 4
Homework 4 will have you extending the code from Homework 3 to integrate personalization and feedback.
1. Implement Rocchio’s method with pseudofeedback for sparse queries in `ranker.py`
2. Implement Rocchio’s method with pseudofeedback for dense queries in `vector_ranker.py`
3. Generate the query results for four systems:
  - A L2R system with a BM25 ranker (like what you used in Homework 2)
  - A L2R system with a BM25 ranker and pseudofeedback
  - A L2R system with a VectorRanker
  - A L2R system with a VectorRanker and pseudofeedback
4. Extend your code to support Personalized PageRank
5. Implement Personalized BM25

## Changes from Assignment 3

### Data Changes
- We will continue using the `wikipedia_200k_dataset.jsonl` for HW4 as our base corpus
- We will have a new file `personalization.jsonl` that contains information on the two simulated users
   - This file contains the documents in the seed set
---

### `document_preprocessor.py`

There are no changes.

---

### `indexing.py`

There are no changes.

---

### `relevance.py`

There are no changes.

---

### `ranker.py`

#### `Ranker` class

- In `query`, there are new arguments to deal with the pseudo-feedback
  - `pseudofeedback_num_docs`: if pseudo-feedback is requested, the number of top-ranked documents to be used in the query, default is 0 (not using pseudo-feedback)
  - `pseudofeedback_alpha`: if pseudo-feedback is used, the alpha parameter for weighting how much to include of the original query in the updated query
  - `pseudofeedback_beta`: if pseudo-feedback is used, the beta parameter for weighting how much to include of the relevant documents in the updated query
  - `user_id`: the integer id of the user who is issuing the query or None if the user is unknown
- Within `query` you must create the pseudodocument from the specified number of pseudo-relevant results

#### `RelevanceScorer` class

- You must implement the `PersonalizedBM25` class
  - Using the formula in the PDF, implement Personalized BM25
  - `__init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                  parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:`
  - `revelant_doc_index`: The inverted index of only documents a user has rated as relevant, which is used when calcuating the personalized part of BM25
  - Think carefully about where the `relevant_doc_index` is used

---

### `l2r.py`
- No major changes
- Make sure to update any uses of the `RelevanceScorer` class

---

### `network_features.py`
- `calculate_page_rank(self, graph, damping_factor=0.85, iterations=100, weights=None) -> list[float]:` now has a `weights` argument for Personalized PageRank
- `weights`: a data structure containing the restart distribution as a vector (over the length of nodes) or a dict {node: weight}
- From your HW3 implementation, should only be one change
  - check https://scikit-network.readthedocs.io/en/latest/reference/ranking.html#pagerank

---

### `vector_ranker.py`

- Similar to the `Ranker` class, the `VectorRanker` class' `query` function also has new arguments for pseudo-feedback
  - `pseudofeedback_num_docs`: If pseudo-feedback is requested, the number of top-ranked documents to be used in the query, default is 0 (not using pseudo-feedback)
  - `pseudofeedback_alpha`: If pseudo-feedback is used, the alpha parameter for weighting how much to include of the original query in the updated query
  - `pseudofeedback_beta`: If pseudo-feedback is used, the beta parameter for weighting how much to include of the relevant documents in the updated query
 
- If using pseudo-feedback, after encoding the query using the biencoder
  -  Get the most-relevant document vectors for the initial query
  -  Compute the average vector of the most relevant docs
  -  Combine the original query doc with the feedback doc to use as the new query embedding
    
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
