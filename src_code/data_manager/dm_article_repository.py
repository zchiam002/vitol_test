from pydantic import BaseModel
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src_code.data_manager.dm_article import DMArticle
from src_code.data_manager.dm_query import DMQuery

class DMArticleRepository(BaseModel):
    repository:List[DMArticle]

    # To return the top n articles of the

    # To intake a query and return a relevance score
    def get_relevance_score (self, incoming_query: str) -> float:
        # Process the incoming query 
        query = DMQuery(content=incoming_query)
        
        # Initialize the basic vectorization of the articles 
        self._initialize_basic_vectorization()

        # Now lets do the comparison 
        relevance_score_df = self._get_basic_top_n_scores(query=query, top_n=1)

        return relevance_score_df['relevance_score'][0]
    
    # To intake an article index and return a novelty score 
    def get_novelty_score (self, article_idx: int) -> float:
        # Initialize the basic vectorization of the articles 
        self._initialize_basic_vectorization()

        self._initialize_basic_novelty_matrix()
        if not (0 <= article_idx < len(self._novelty_matrix)):
            raise IndexError(f"Index {article_idx} is out of bounds for the document repository.")
        return self._novelty_matrix[article_idx]
    
    # To get the weights associated with the basic method of calculating relevance 
    def _get_basic_calculation_weights (self) -> Dict[str, float]:
        WEIGHTS = {
            'category': 2.0,
            'title': 1.5,
            'content': 1.0
        }
        return WEIGHTS

    # To do basic computation of the relevance score 
    def _get_basic_top_n_scores (self, query: DMQuery, top_n: int) -> pd.DataFrame: 
        # Definition of the weights associated with the category, title and content 
        WEIGHTS = self._get_basic_calculation_weights()
        denominator = sum(WEIGHTS.values())
        
        # Transform the incoming query using each of the three vectorizers.
        query.basic_preprocess()

        query_vector_category = self._category_vectorizer.transform([query.content])
        query_vector_title = self._title_vectorizer.transform([query.content])
        query_vector_content = self._content_vectorizer.transform([query.content])

        final_scores = (
            WEIGHTS['category'] / denominator * self._get_basic_similarity(query_vector_category, self._category_tfidf_matrix) +
            WEIGHTS['title'] / denominator * self._get_basic_similarity(query_vector_title, self._title_tfidf_matrix) +
            WEIGHTS['content'] / denominator * self._get_basic_similarity(query_vector_content, self._content_tfidf_matrix)
        )

        # Get the indices of the top N documents based on the final weighted scores.
        top_n_indices = np.argsort(final_scores)[::-1][:top_n]

        results_list = []
        for rank, idx in enumerate(top_n_indices):
            most_similar_document = self.repository[idx]
            
            results_list.append({
                'rank': rank + 1,
                'title': most_similar_document.title,
                'relevance_score': final_scores[idx]
            })

        # Create the pandas DataFrame from the results list.
        results_df = pd.DataFrame(results_list)

        return results_df
    
    # To do basic similarity calculation
    def _get_basic_similarity (self, vector_1: csr_matrix, vector_2: csr_matrix) -> float:
        return cosine_similarity(vector_1, vector_2).flatten()

    # To do basic novelty calculation 
    def _initialize_basic_novelty_matrix(self) -> None:
        if not hasattr(self, '_novelty_matrix'):
            num_docs = len(self.repository)
            novelty_scores = np.zeros(num_docs)
            
            # Weights for novelty calculation. Content is weighted higher as it is
            # the best source of unique information.
            NOVELTY_WEIGHTS = self._get_basic_calculation_weights()
            denominator = sum(NOVELTY_WEIGHTS.values())
            
            # Calculate pair-wise similarities for each field.
            category_similarities = cosine_similarity(self._category_tfidf_matrix, self._category_tfidf_matrix)
            title_similarities = cosine_similarity(self._title_tfidf_matrix, self._title_tfidf_matrix)
            content_similarities = cosine_similarity(self._content_tfidf_matrix, self._content_tfidf_matrix)
            
            for i in range(num_docs):
                # The diagonal of the similarity matrix is always 1.0 (similarity to self).
                # We must exclude this from the average.
                avg_category_similarity = (np.sum(category_similarities[i]) - 1.0) / (num_docs - 1)
                avg_title_similarity = (np.sum(title_similarities[i]) - 1.0) / (num_docs - 1)
                avg_content_similarity = (np.sum(content_similarities[i]) - 1.0) / (num_docs - 1)
                
                # Combine the average similarities with the novelty weights.
                weighted_avg_similarity = (
                    (NOVELTY_WEIGHTS['category'] / denominator) * avg_category_similarity +
                    (NOVELTY_WEIGHTS['title'] / denominator) * avg_title_similarity +
                    (NOVELTY_WEIGHTS['content'] / denominator) * avg_content_similarity
                )
                
                novelty_scores[i] = 1.0 - weighted_avg_similarity

            self._novelty_matrix = novelty_scores
        return 

    # To do basic vectorization of the articles 
    def _initialize_basic_vectorization(self) -> None:    
        attributes = ['category', 'title', 'content']
        
        for attribute in attributes:
            vectorizer_attr = f'_{attribute}_vectorizer'
            matrix_attr = f'_{attribute}_tfidf_matrix'
            
            # Check if the vectorizer has already been created
            if not hasattr(self, vectorizer_attr):
                vectorizer = TfidfVectorizer(stop_words='english')
                
                # Get the list of strings for the current attribute
                data_to_vectorize = self._get_consolidate_into_list(attribute=attribute)
                
                # Fit and transform the data
                tfidf_matrix = vectorizer.fit_transform(data_to_vectorize)
                
                # Assign the vectorizer and matrix to the instance
                setattr(self, vectorizer_attr, vectorizer)
                setattr(self, matrix_attr, tfidf_matrix)

        return

    # To consolidate all the content in the repository into a single list 
    def _get_consolidate_into_list(self, attribute: str) -> List[str]:
        # Define a set of valid attributes to prevent arbitrary attribute access
        valid_attributes = {'category', 'title', 'content'}

        # Check if the requested attribute is valid
        if attribute not in valid_attributes:
            raise ValueError(f"Invalid attribute: '{attribute}'. Valid attributes are {valid_attributes}.")

        # Use a single list comprehension with getattr()
        return [getattr(article, attribute) for article in self.repository]