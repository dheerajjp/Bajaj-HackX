from fastapi import APIRouter, UploadFile
import tempfile
from services import parser, embedding, pinecone_service

router = APIRouter()

@router.post("/")
async def ingest_document(file: UploadFile):
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name

    # Parse
    if file.filename.endswith(".pdf"):
        text = parser.parse_pdf(file_path)
    elif file.filename.endswith(".docx"):
        text = parser.parse_docx(file_path)
    else:
        return {"error": "Unsupported file type"}

    
    chunks = parser.chunk_text(text)

    
    embeddings = embedding.embed_text(chunks)

    
    pinecone_service.upsert_embeddings(chunks, embeddings, metadata={"document_id": file.filename})

    return {"message": "Document processed successfully"}
