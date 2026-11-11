import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()
oa, pc = OpenAI(), Pinecone()
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "checkpoint-rag"))
print(index.describe_index_stats())

for q in [
    "What's the maximum number of steps allowed in a single workflow?",
    "Users get stuck in an SSO redirect loop after switching to Azure AD. What's the fix?",
]:
    vec = oa.embeddings.create(model="text-embedding-3-small", input=q).data[0].embedding
    res = index.query(vector=vec, top_k=3, include_metadata=True, namespace="v1")
    print("\nQ:", q)
    for m in res.matches:
        print(f"  {m.score:.3f} | {m.id} | {m.metadata.get('section') or m.metadata.get('status')}")