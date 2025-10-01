from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from langchain_core.runnables import RunnableBranch,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

class Response(BaseModel):
    sentiment:Literal["positive","negitive"] = Field(description="the sentiment of the review classified either negitive or positive")

parser=PydanticOutputParser(pydantic_object=Response)

prompt1=PromptTemplate(template="classify the given review's sentiment into postive or negitive \n {review} \n format:{format_instructions}",input_variables=["review"] ,partial_variables={"format_instructions":parser.get_format_instructions()})


prompt2=PromptTemplate(template="give the appropriate response to this negitive review for the customer. it should be short and engaging for the customer \n {review}\n ",input_variables=["review"] )

prompt3=PromptTemplate(template="give the appropriate response to this postive review or the customer. it should be short and engaging for the customer \n {review}\n ",input_variables=["review"] )

chain1= prompt1 | model | parser

parser2=StrOutputParser()

branchChain=RunnableBranch(
    (lambda x:x.sentiment=="positive", prompt3 | model|parser2),
    (lambda x:x.sentiment=="negitive", prompt2 | model |parser2),
RunnableLambda(lambda x:"coudnt find the sentiment")
)


# wrong the review was not sent to prompt 2 and 3
finalChain= chain1|branchChain

result=finalChain.invoke("the smartphone is too good")

print(result)