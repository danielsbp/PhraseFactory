from typing import Union
from pydantic import BaseModel
class Category(BaseModel):
    name : str
    description : Union[str, None] = None
    image : str    
    icon : str