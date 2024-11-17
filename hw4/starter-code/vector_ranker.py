from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not 
        # submit that code to the autograder
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name) # Initialize the bi-encoder model here
        
        # TODO: Also include other necessary initialization code
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid
        self.docid_to_row = {docid: i for i, docid in enumerate(row_to_docid)}

    def query(self, query: str, pseudofeedback_num_docs=0,
              pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: We don't use the user_id parameter in vector ranker. It is here just to align all the
                    Ranker interfaces.

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        pass
        # NOTE: Do not forget to handle edge cases on the input
        # TODO: Encode the query using the bi-encoder
        if not query.strip():
            return []
        query_embedding = self.biencoder_model.encode(query)
        similarity_scores = util.dot_score(query_embedding, self.encoded_docs)[0].cpu().numpy()
        docid_score_pairs = [(self.row_to_docid[i], score) for i, score in enumerate(similarity_scores)]
        sorted_docid_score_pairs = sorted(docid_score_pairs, key=lambda x: x[1], reverse=True)
        
        if pseudofeedback_num_docs > 0:
        # TODO (HW4): If the user has indicated we should use feedback, then update the
        #  query vector with respect to the specified number of most-relevant documents

            # TODO (HW4): Get the most-relevant document vectors for the initial query
            # TODO (HW4): Compute the average vector of the specified number of most-relevant docs
            #  according to how many are to be used for pseudofeedback
            # TODO (HW4): Combine the original query doc with the feedback doc to use
            #  as the new query embedding
            pseudo_rel_docs = [docid for docid, _ in sorted_docid_score_pairs[:pseudofeedback_num_docs]]
            pseudo_vec = np.mean([self.encoded_docs[self.docid_to_row[docid]] for docid in pseudo_rel_docs], axis=0)
            augmented_query_embedding = pseudofeedback_alpha * query_embedding + pseudofeedback_beta * pseudo_vec
            # print("complete pseudo-feedback, start quering")

            similarity_scores = util.dot_score(augmented_query_embedding, self.encoded_docs)[0].cpu().numpy()
            docid_score_pairs = [(self.row_to_docid[i], score) for i, score in enumerate(similarity_scores)]
            sorted_docid_score_pairs = sorted(docid_score_pairs, key=lambda x: x[1], reverse=True)
            return sorted_docid_score_pairs
        
        else:
        # TODO: Score the similarity of the query vec and document vectors for relevance
        # TODO: Generate the ordered list of (document id, score) tuples
        # TODO: Sort the list by relevance score in descending order (most relevant first)
            return sorted_docid_score_pairs