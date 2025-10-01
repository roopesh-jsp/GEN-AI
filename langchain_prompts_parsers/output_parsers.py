from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str = Field(description="name of the person")
    city:str = Field(description="city of the person")
    age:int = Field(description="age of the person")

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(template='generate fake fictional details name,age and city of a person living in city {place} \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# both worked for me 
# template=PromptTemplate(template='generate fake fictional details  of a person living in city {place} \n {format_instructions}',
#     input_variables=['place'],
#     partial_variables={'format_instructions':parser.get_format_instructions()}
# )


chain = template | model | parser

final_result= chain.invoke({
    'place':"andhra pradesh"
})

print(final_result)


