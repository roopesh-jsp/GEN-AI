from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# requires HUGGINGFACEHUB_API_TOKEN in your environment
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # must be a chat/instruct model
    task="text-generation",  # or "text2text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)


model =ChatHuggingFace(llm=llm)

result=model.invoke("what is capital of india")

print(result.content)


