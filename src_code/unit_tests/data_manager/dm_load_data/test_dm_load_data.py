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

        assert len(articles.repository) == 2

        article_categories = []
        article_titles = []
        for article in articles.repository: 
            article_categories.append(article.category)
            article_titles.append(article.title)

        assert article_categories == ['business', 'business']
        assert article_titles == ['Dollar gains on Greenspan speech', 'Ad sales boost Time Warner profit']

        return 

    def test_to_dataframe (self):
        articles = DMLoadData.load_from_directory(self.data_dir)
        df = DMLoadData.to_dataframe(articles.repository)
        assert df['category'][0] == 'business'
        assert df['title'][0] == 'Dollar gains on Greenspan speech'
        assert 'The dollar has hit its' in df['content'][0]

        return 
    
    def test_preprocess_text (self):
        articles = DMLoadData.load_from_directory(self.data_dir)
        for article in articles.repository: 
            article.basic_preprocess()

        assert articles.repository[0].category == 'business'
        assert articles.repository[0].title == 'dollar gain greenspan speech' 
        assert 'dollar hit highest level euro almost three' in articles.repository[0].content    

        return 
    
if __name__ == "__main__":
    unittest.main()
