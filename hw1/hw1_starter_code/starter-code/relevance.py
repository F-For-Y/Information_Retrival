

"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (marginally relevant) and 2 (very relevant). When you calculate MAP, treat 1s and 2s are relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant. Thus, we have 
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""
import math
import pandas as pd
from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF, YourRanker
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    rel = 0
    for score in search_result_relevances[:cut_off]:
        if score == 1:
            rel += 1
    return rel / len(search_result_relevances[:cut_off])
    
    


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_of: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    rel_dcg = 0
    rel_idcg = 0
    search_result_relevances = search_result_relevances[:cut_of]
    ideal_relevance_score_ordering = ideal_relevance_score_ordering[:cut_of]
    # Implement DCG
    for i, score in enumerate(search_result_relevances):
        rel_dcg += (2**score - 1) / (math.log2(i + 2))

    # Implement IDCG
    for i, score in enumerate(ideal_relevance_score_ordering):
        rel_idcg += (2**score - 1) / (math.log2(i + 2))
        
    return rel_dcg / rel_idcg


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance csv dataset
    # column query, title, docid, rel
    df = pd.read_csv(relevance_data_filename)
    val_dict = {}
    for i in range(len(df)):
        val_dict.setdefault(df['query'][i], {})
        val_dict[df['query'][i]][df['docid'][i]] = df['rel'][i]
        
    # create the index with full document collection: wikipedia_200k_dataset.jsonl
    # TODO: Run each of the dataset's queries through your ranking function
    query_result = {}
    for query in val_dict.keys():
        ranked_doc = ranker.query(query)   
        query_result.setdefault(query, {})
        query_result[query]['docid'] = [item[0] for item in ranked_doc]
        map_lable = []
        ndcg_label = []
        for item in ranked_doc:
            if item[0] in val_dict[query]:
                map_lable.append(1 if val_dict[query][item[0]] > 3 else 0)
                ndcg_label.append(val_dict[query][item[0]])
            else:
                ndcg_label.append(0)
        query_result[query]['map_label'] = map_lable
        query_result[query]['idcg_label'] = ndcg_label
    
    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out\
    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.
    # NOTE: NDCG can use any scoring range, so no conversion is needed.
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    for query in query_result.keys():
        query_result[query]['map'] = map_score(query_result[query]['map_label'])
        query_result[query]['ndcg'] = ndcg_score(query_result[query]['idcg_label'], sorted(query_result[query]['idcg_label'], reverse=True))
    
    return {'map': 0, 'ndcg': 0, 'map_list': [], 'ndcg_list': []}


if __name__ == '__main__':
    pass
