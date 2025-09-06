import re
from typing import Optional
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel

class DMArticle(BaseModel):
    '''The category is defined by the folder name,
    the title is extracted from the first line of the article,
    the rest are the contents.
    '''
    category: str
    title: str
    content: str

    original_category: Optional[str] = None
    original_title: Optional[str] = None
    original_content: Optional[str] = None

    # Basic preprocessing of the data 
    def basic_preprocess (self, cache_original: bool = True) -> None:

        def preprocess_text(text: str) -> str:
            text = text.lower()
            # Remove Punctuation and Special Characters using regex
            text = re.sub(r'[^a-z\s]', '', text)

            # Tokenization & Stop Word Removal
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in text.split() if word not in stop_words]

            # 5. Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # Join the tokens back into a single string
            return " ".join(lemmatized_tokens)

        if cache_original:
            self.original_category = self.category
            self.original_title = self.title
            self.original_content = self.content

        self.category = preprocess_text(self.category)
        self.title = preprocess_text(self.title)
        self.content = preprocess_text(self.content)

        return 
    
    # def get_key_terms(self, n: int = 5) -> List[str]:
    #     """
    #     Returns a list of the most frequent words in the article's content,
    #     which serve as key terms.

    #     Args:
    #         n (int): The number of key terms to return.

    #     Returns:
    #         List[str]: A list of the top n most frequent words.
    #     """
    #     # The content is already preprocessed and in a single string.
    #     words = self.content.split()
        
    #     # Use Counter to get the frequency of each word
    #     word_counts = Counter(words)
        
    #     # Return the top n most common words
    #     most_common_words = word_counts.most_common(n)
        
    #     return [word for word, count in most_common_words]