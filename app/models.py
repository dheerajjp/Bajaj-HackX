
from typing import List, Optional, Any, Dict
from pydantic import BaseModel

class RunRequest(BaseModel):
    documents: Any  # Can be str (single URL) or List[str]
    questions: List[str]

class Answer(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    source_clauses: Optional[list] = None  # list of dicts

class RunResponse(BaseModel):
    answers: List[str]  # keep to exact spec expected by judge
    # Optional debug output if user adds ?debug=true
    traces: Optional[List[Answer]] = None
