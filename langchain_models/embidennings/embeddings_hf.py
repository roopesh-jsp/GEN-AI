from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# llm=HuggingFaceEndpoint(repo_id="sentence-transformers/all-MiniLM-L6-v2",task="")

embeding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

result=embeding.embed_query("hai all how are you")

print(str(result))

# from sentence_transformers import SentenceTransformer
# sentences = ["This is an example sentence", "Each sentence is converted"]

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddings = model.encode(sentences)
# print(embeddings)
