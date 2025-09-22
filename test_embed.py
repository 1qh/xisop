from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

r = OpenAI().embeddings.create(
  model='embeddinggemma:300m-qat-q4_0',
  input=['Hello world', 'Hello there', 'Hello you'],
)

print(r)
