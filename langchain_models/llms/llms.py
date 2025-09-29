from langchain_google_genai  import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model= GoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.7)
 
result=model.invoke("what is capital of india")

print(result)

