from typing import List, Optional
from pydantic import BaseModel
from src_code.data_manager.dm_article_repository import DMArticleRepository
from src_code.data_manager.dm_article import DMArticle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class DMUserProfile(BaseModel):
    user_id: str
    article_history: Optional[List[int]] = []
    history_limit: int

    def append_article (self, article_idx: int) -> None:
        # We use a first-in, first-out method to update the article history 
        curr_history_len = len(self.article_history)

        if article_idx not in self.article_history: 
            if curr_history_len < self.history_limit: 
                self.article_history.append(article_idx)
            else: 
                self.article_history=self.article_history[1:] + [article_idx]

        return 
    
    def get_interest_vector (self, article_repository: DMArticleRepository, top_n_key_words: int=5) -> csr_matrix:
        # Get all the relevant articles 
        relevant_articles = [article_repository.get_article(article_idx=i) for i in self.article_history]

        # For each relevant article, get the key words 
        key_word_list = []
        for article in relevant_articles:
            # Assuming article has a get_key_terms() method
            key_word_list.extend(article.get_key_terms(top_n=top_n_key_words))
        
        # Join all keywords into a single string for vectorization
        combined_keywords = " ".join(key_word_list)
        
        # Vectorize the key word list 
        article_repository.build_repository_tfidf_matrix(top_n_key_words=top_n_key_words)
        user_key_word_vector = article_repository.repository_vectorizer.transform([combined_keywords])

        return user_key_word_vector

    

    




    

