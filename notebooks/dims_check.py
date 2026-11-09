from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                 # puts OPENAI_API_KEY into the environment
client = OpenAI()             # the SDK finds the key automatically

texts = [
    "cat",
    "What's the maximum number of steps allowed in a single workflow?",
]

resp = client.embeddings.create(model="text-embedding-3-small", input=texts)

for text, item in zip(texts, resp.data):
    print(f"{len(item.embedding)} dims | first 5: {[round(x, 4) for x in item.embedding[:5]]} | {text[:40]}")

print("Tokens billed:", resp.usage.total_tokens)