import unittest
import os 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..', '')) + '//'
test_dir = root_dir + 'src_code//unit_tests//data_manager//dm_load_data//'
import sys 
sys.path.append(root_dir)

from src_code.data_manager.dm_load_data import DMLoadData

class TestArticleLoader(unittest.TestCase):
    def setUp(self):
        self.data_dir = test_dir + 'input_data//'

    def test_load_data(self):
        articles = DMLoadData.load_from_directory(self.data_dir)

        assert len(articles) == 2

        article_categories = []
        article_titles = []
        for article in articles: 
            article_categories.append(article.category)
            article_titles.append(article.title)

        assert article_categories == ['business', 'business']
        assert article_titles == ['Dollar gains on Greenspan speech', 'Ad sales boost Time Warner profit']

        return 

    def test_to_dataframe (self):
        articles = DMLoadData.load_from_directory(self.data_dir)
        df = DMLoadData.to_dataframe(articles)

        assert list(df.columns) == ['category', 'title', 'content']
        assert df['category'][0] == 'business'
        assert df['title'][0] == 'Dollar gains on Greenspan speech'
        assert 'The dollar has hit its' in df['content'][0]

        return 
    
if __name__ == "__main__":
    unittest.main()
