import numpy as np
from sknetwork.ranking import PageRank, HITS
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm
from sknetwork.data import from_edge_list
import gzip


class NetworkFeatures:
    """
    A class to help generate network features such as PageRank scores, HITS hub score and HITS authority scores.
    This class uses the scikit-network library https://scikit-network.readthedocs.io to calculate node ranking values.

    OPTIONAL reads
        1. PageRank: https://towardsdatascience.com/pagerank-algorithm-fully-explained-dc794184b4af
        2. HITS: https://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture4/lecture4.html
    """
    def load_network(self, network_filename: str, total_edges: int):
        """
        Loads the network from the specified file and returns the network. A network file 
        can be listed using a .csv or a .csv.gz file.

        Args:
            network_filename: The name of a .csv or .csv.gz file containing an edge list
            total_edges: The total number of edges in an edge list

        Returns:
            The loaded network from sknetwork
        """
        # NOTE: there are 92650947 edges in the big network we give you. However,
        # do not hard code this value here, as it will cause the auto-grader tests
        # to break

        # NOTE: Trying to create the network from a pandas dataframe will not work 
        # (too much memory). You'll need to read the documentation to figure out how to 
        # load in the network in the most memory-efficient way possible. This is the 
        # "hard part" of this class's implementation as it requires you to think about
        # memory and data representations. 

        # NOTE: your code should support reading both gzip and non-gzip formats

        # NOTE: On a reference laptop, loading the network file's data took ~90 seconds
        # and constructing the network took ~75 seconds. We estimate that the entire 
        # network construction memory requirement is under 5GB based on tests with 
        # the reference implementation.
        edge_list = []

        if network_filename.endswith('.gz'):
            f = gzip.open(network_filename, 'rt')
        else:
            f = open(network_filename, 'r')
            
        next(f)
        for line in tqdm(f):
            edge_list.append(tuple(map(int, line.strip().split(','))))
        assert len(edge_list) == total_edges, f"Expected {total_edges} edges, got {len(edge_list)} edges"
            
        f.close()
        # return the csr_matrix
        return from_edge_list(edge_list, directed=True, reindex=True)
        
    def calculate_page_rank(self, graph, damping_factor=0.85, iterations=100, weights=None) -> list[float]:
        """
        Calculates the PageRank scores for the provided network and
        returns the PageRank values for all nodes.

        Args:
            graph: A graph from sknetwork
            damping_factor: The complement of the teleport probability for the random walker
                For example, a damping factor of .8 has a .2 probability of jumping after each step.
            iterations: The maximum number of iterations to run when computing PageRank
            weights: if Personalized PageRank is used, a data structure containing the restart distribution
                     as a vector (over the length of nodes) or a dict {node: weight}

        Returns:
            The PageRank scores for all nodes in the network (array-like)
        
        TODO (hw4): Note that `weights` is added as a parameter to this function for Personalized PageRank.
        """
        # TODO (HW4): Use scikit-network to calculate and return PageRank scores; if the user has indicated
        #  we should use Personalized PageRank, return the scores using the given weights
        pr = PageRank(damping_factor=damping_factor, n_iter=iterations)
        
        if weights is not None:
            n_nodes = graph.adjacency.shape[0]

            if isinstance(weights, dict):
                # Convert the dict to an array
                personalization = np.zeros(n_nodes)
                for node_index, weight in weights.items():
                    personalization[node_index] = weight
            else:
                # Assume weights is array-like of length n_nodes
                personalization = np.array(weights)

            # Normalize the personalization vector
            sum_weights = personalization.sum()
            if sum_weights > 0:
                personalization /= sum_weights
            else:
                raise ValueError("The sum of weights must be greater than 0.")

            pr.fit(graph.adjacency, weights=personalization)
        else:
            pr.fit(graph.adjacency)

        return list(pr.scores_)
        # return list(pr.fit(graph.adjacency).scores_)
        

    def calculate_hits(self, graph) -> tuple[list[float], list[float]]:
        """
        Calculates the hub scores and authority scores using the HITS algorithm
        for the provided network and returns the two lists of scores as a tuple.

        Args:
            graph: A graph from sknetwork

        Returns:
            The hub scores and authority scores (in that order) for all nodes in the network
        """
        # TODO: Use scikit-network to run HITS and return HITS hub scores and authority scores
        # NOTE: When returning the HITS scores, the returned tuple should have the hub scores in index 0 and
        #       authority score in index 1
        hits = HITS()
        hits.fit(graph.adjacency)
        hubs = list(hits.scores_row_)
        authorities = list(hits.scores_col_)
        return (hubs, authorities)

    def get_all_network_statistics(self, graph, weight=None) -> DataFrame:
        """
        Calculates the PageRank and the hub scores and authority scores using the HITS algorithm
        for the provided network and returns a pandas DataFrame with columns: 
        'docid', 'pagerank', 'authority_score', and 'hub_score' containing the relevant values
        for all nodes in the network.

        Args:
            graph: A graph from sknetwork

        Returns:
            A pandas DataFrame with columns 'docid', 'pagerank', 'authority_score', and 'hub_score'
            containing the relevant values for all nodes in the network
        """

        # TODO: Calculate all the Pagerank and HITS scores for the network graph and store it in a dataframe

        # NOTE: We use a DataFrame here for efficient storage of the values on disk.
        # However, when you actually use these values, you'll convert this DataFrame
        # to another dictionary-based representation for faster lookup when making
        # the L2R features.

        # NOTE: Return the dataframe and save the dataframe as a CSV or JSON
        pr_scores = self.calculate_page_rank(graph, weights=weight)
        hub_scores, authority_scores = self.calculate_hits(graph)
        
        df = DataFrame({'docid': list(graph.names), 'pagerank': pr_scores, 'authority_score': authority_scores, 'hub_score': hub_scores})
        return df


# Example main function
if __name__ == '__main__':
    nf = NetworkFeatures()
    g = nf.load_network('edgelist.csv', 92650947)
    final_df = nf.get_all_network_statistics(g)
    final_df.to_csv('network_stats.csv', index=False)
