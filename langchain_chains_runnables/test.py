from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

print(llm.invoke("2+2=?").content)

