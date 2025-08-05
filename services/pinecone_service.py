import pinecone
import os

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
index = pinecone.Index("policy-clauses")

def upsert_embeddings(chunks, embeddings, metadata):
    vectors = []
    for i, embedding in enumerate(embeddings):
        vectors.append((
            f"{metadata['document_id']}_{i}",
            embedding,
            {"text": chunks[i], **metadata}
        ))
    index.upsert(vectors=vectors)
