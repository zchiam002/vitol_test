import os
import pandas as pd
from pydantic import ValidationError
from typing import List

from src_code.data_manager.dm_article import DMArticle

class DMLoadData: 
    @staticmethod
    def load_from_directory(base_path: str) -> List[DMArticle]:
        articles: List[DMArticle] = []
        
        if not os.path.exists(base_path):
            print(f"Error: The provided path '{base_path}' does not exist.")
            return articles

        # List all subdirectories (categories) within the base path
        try:
            categories = [d for d in os.listdir(base_path) 
                          if os.path.isdir(os.path.join(base_path, d))]
        except OSError as e:
            print(f"Error listing directories in '{base_path}': {e}")
            return articles

        for category in categories:
            category_path = os.path.join(base_path, category)
            
            # List all files in the current category directory
            for filename in os.listdir(category_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(category_path, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # The first line is the title, the rest is the content
                        lines = content.split('\n', 1)
                        title = lines[0].strip()
                        article_content = lines[1].strip() if len(lines) > 1 else ""

                        try:
                            article = DMArticle(
                                category=category,
                                title=title,
                                content=article_content
                            )
                            articles.append(article)
                        except ValidationError as e:
                            print(f"Validation error for file '{file_path}': {e}")

                    except Exception as e:
                        print(f"Error processing file '{file_path}': {e}")
        
        return articles

    @staticmethod
    def to_dataframe(articles: List[DMArticle]) -> pd.DataFrame:
        data = [a.model_dump() for a in articles]
        return pd.DataFrame(data)



