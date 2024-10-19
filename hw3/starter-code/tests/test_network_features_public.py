import unittest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# modules being tested
from network_features import NetworkFeatures

class TestNetworkFeatures(unittest.TestCase):

    def test_load_network(self):
        '''Test loading the network using load_network (with CSV file)'''

        nf = NetworkFeatures()

        network = nf.load_network('toy-network.csv', 80)

        self.assertEqual(len(network.names), 20, 'Expected 20 nodes in graph')

    def test_load_network_gz(self):
        '''Test loading the network using load_network (with GZ file)'''

        nf = NetworkFeatures()

        network = nf.load_network('toy-network.csv.gz', 80)

        self.assertEqual(len(network.names), 20, 'Expected 20 nodes in graph')

    def test_calculate_page_rank(self):
        '''Test calculating the Pagerank scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)
        
        pr_scores = [0.05125961, 0.05951749, 0.04892111, 0.05118604, 0.08388691,
                     0.05019047, 0.06019026, 0.06067016, 0.0379761 , 0.02879423,
                     0.0707906 , 0.05905573, 0.04016703, 0.06474813, 0.02881887,
                     0.04909074, 0.01851644, 0.02785495, 0.04900831, 0.05935682 ]
        
        est_pr_scores = list(nf.calculate_page_rank(network))

        for pr, est_pr in zip(pr_scores, est_pr_scores):
            self.assertAlmostEqual(est_pr, pr, places=3,
                                   msg='PageRank scores do not match')
        

    def test_calculate_hits(self):
        '''Test calculating the HITS scores'''

        nf = NetworkFeatures()
        network = nf.load_network('toy-network.csv', 80)

        actual_hub_scores, actual_authority_scores = nf.calculate_hits(network)

        # hub == auth for the toy network
        expected_scores = [0.16546851, 0.27122355, 0.23465519, 0.18502592, 0.32562661,
                      0.19676417, 0.29508263, 0.25060985, 0.19212819, 0.1095624,
                      0.31498321, 0.27880198, 0.12209237, 0.24641294, 0.09718952,
                      0.23205709, 0.05497426, 0.14291885, 0.2388066, 0.26433229]
        
        est_hub_scores, est_auth_scores = nf.calculate_hits(network)
        est_hub_scores = list(est_hub_scores)
        est_auth_scores = list(est_auth_scores)

        for hub, est_hub, est_auth in zip(expected_scores, est_hub_scores, est_auth_scores):
            self.assertAlmostEqual(est_hub, hub, places=3,
                                   msg='Hub scores do not match')
            self.assertAlmostEqual(est_auth, hub, places=3,
                                   msg='Auh scores do not match')
        
    def test_get_all_network_statistics(self):
        nf = NetworkFeatures()
        expected_df = pd.read_csv('network_stats.csv')
        network = nf.load_network('toy-network.csv', 80)
        actual_df = nf.get_all_network_statistics(network)
        pd.testing.assert_frame_equal(actual_df, expected_df, check_dtype=False, check_exact=False)



if __name__ == '__main__':
    unittest.main()
