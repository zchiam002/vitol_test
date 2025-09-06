import re
from typing import Optional
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pydantic import BaseModel

class DMQuery(BaseModel):
    content: str
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
            self.original_content = self.content

        self.content = preprocess_text(self.content)

        return 