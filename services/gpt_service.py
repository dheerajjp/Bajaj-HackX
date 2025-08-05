from openai import OpenAI
client = OpenAI()

def answer_question(question, context):
    prompt = f"""
You are an insurance policy expert.
Answer the question below using only the provided context.
Return JSON with "answer", "rationale", and "supporting_clauses".

Question: {question}

Context:
{context}
"""
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    import json
    return json.loads(completion.choices[0].message.content)
