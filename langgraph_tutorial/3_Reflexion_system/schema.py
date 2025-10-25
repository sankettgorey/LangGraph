from pydantic import BaseModel, Field

from typing import List, Optional


class Reflection(BaseModel):

    missing: str=Field(description='critique of what is missing')
    superfluous: str=Field(description="critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """answer the question"""

    answer: str=Field(description="~250 word detailed answer of the question")
    search_queries: List[str]=Field(description='1-3 search queries for researching improvements to address ceitique of your current answer')
    reflection: Reflection=Field(description="Your reflection on initial answer")



class RevisorTemplate(AnswerQuestion):
    """revise your original answer to the given question"""

    references: List[str] = Field(description= 'citations motivating your revised answer')