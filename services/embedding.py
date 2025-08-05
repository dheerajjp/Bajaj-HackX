from openai import OpenAI

client = OpenAI()

def embed_text(texts):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [d.embedding for d in response.data]
