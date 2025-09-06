import unittest
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', '')) + '//'
test_dir = root_dir + 'src_code//unit_tests//data_manager//dm_article_repository//'
import sys 
sys.path.append(root_dir)
from src_code.data_manager.dm_load_data import DMLoadData

class TestArticleRepository(unittest.TestCase):
    def setUp(self):
        self.data_dir = test_dir + 'input_data//'   

    def test_get_relevance_score (self):
        article_repository = DMLoadData.load_from_directory(self.data_dir)
        relevance_score = article_repository.get_relevance_score('internet business profit')
        assert round(relevance_score, 2) == 0.64

        relevance_score = article_repository.get_relevance_score('charlie brown peanuts')
        assert round(relevance_score, 2) == 0.00

        return 
    
    def test_get_novelty_score (self):
        article_repository = DMLoadData.load_from_directory(self.data_dir)
        novelty_score = article_repository.get_novelty_score(0)
        assert round(novelty_score, 2) == 0.85

        return

if __name__ == "__main__":
    unittest.main()
