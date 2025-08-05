from fastapi import APIRouter
from pydantic import BaseModel
from services import embedding, pinecone_service, gpt_service

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/")
async def query_document(req: QueryRequest):
    
    query_embedding = embedding.embed_text([req.question])[0]

   
    matches = pinecone_service.index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    
    context = "\n\n".join([m.metadata["text"] for m in matches.matches])

    
    answer_json = gpt_service.answer_question(req.question, context)

    return answer_json
