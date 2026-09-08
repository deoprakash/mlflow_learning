'''This file answers:
What does my API accept?'''

from pydantic import BaseModel

class IrisInput(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width: float