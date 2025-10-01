from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

prompt1=PromptTemplate(template="give a small and short joke on this topic :{topic} ",input_variables=["topic"])

praser=StrOutputParser()

chain1= prompt1 | model | praser

parallelChain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "words":RunnableLambda(lambda x:len(x.split(" ")))
}
)

finalChain= chain1 | parallelChain

result=finalChain.invoke({
    "topic":"AI"
})

print(result)