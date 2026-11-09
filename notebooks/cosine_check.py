import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

question = "What's the maximum number of steps allowed in a single workflow?"
candidates = [
    "Workflows are limited to 100 steps.",
    "Is there a cap on workflow length?",
    "How do I update my billing card?",
]

resp = client.embeddings.create(model="text-embedding-3-small", input=[question] + candidates)
vectors = [item.embedding for item in resp.data]
q_vec, cand_vecs = vectors[0], vectors[1:]

print(f"Length of question vector: {math.sqrt(sum(x*x for x in q_vec)):.4f}")   # expect ~1.0

scored = sorted(zip(candidates, cand_vecs), key=lambda p: cosine(q_vec, p[1]), reverse=True)
for text, vec in scored:
    print(f"{cosine(q_vec, vec):.3f} | {text}")