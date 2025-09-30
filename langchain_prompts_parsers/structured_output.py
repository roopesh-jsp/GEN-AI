from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

class quizData(BaseModel):
    question:str = Field(description="this is the question of the quiz")
    options:list[str] = Field(description="these are list of the options for the question above")
    correctAnswer:int = Field(description="this is the index number of the correct answer form the above list of options")

structured_model=llm.with_structured_output(quizData)

result=structured_model.invoke("generate a quiz question for a topic on operating systems")

print(result)

# output:- question='What is the primary function of an operating system?' options=['To manage hardware and software resources.', 'To provide internet access.', 'To create documents.', 'To play games.'] correctAnswer=0