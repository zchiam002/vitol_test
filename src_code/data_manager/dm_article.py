from pydantic import BaseModel

class DMArticle(BaseModel):
    '''The category is defined by the folder name,
    the title is extracted from the first line of the article,
    the rest are the contents.
    '''
    category: str
    title: str
    content: str