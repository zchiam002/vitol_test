import re
from typing import List, Optional
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

class DMArticle(BaseModel):
    '''The category is defined by the folder name,
    the title is extracted from the first line of the article,
    the rest are the contents.
    '''
    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    category: str
    title: str
    content: str

    original_category: Optional[str] = None
    original_title: Optional[str] = None
    original_content: Optional[str] = None

    key_words: Optional[str] = []
    key_word_vector: Optional[csr_matrix] = None

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
    
    def get_key_terms(self, top_n: int = 5) -> List[str]:
        if len(self.key_words) != top_n: 
            # Combine the contents into a single string and process them
            words = self.combine_fields().split()
            
            # Count the frequency of each word
            word_counts = Counter(words)
            
            # Return the top n most common words
            most_common_words = word_counts.most_common(top_n)

            # Generating the return response
            ret_list = [word for word, _ in most_common_words]

            # Updating the key words list and the key word vector 
            self.key_words = ret_list
            vectorizer = TfidfVectorizer(stop_words='english')

            self.key_word_vector = vectorizer.fit_transform(self.key_words)

        return self.key_words
    
    def combine_fields(self) -> str:
        combined_text = f"{self.category} {self.title} {self.content}"
        return combined_text