from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import streamlit as st
load_dotenv()


model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

messages=[
    SystemMessage(content="you are a helpfull math tutor for kids"),
    HumanMessage(content="plaese tell me what is 2+2") #prompt can go here
]

reault= model.invoke(messages)

messages.append(
    AIMessage(content=reault.content)
)

print(messages)


# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# import streamlit as st
# load_dotenv()

# template=PromptTemplate(
#     template="""
# generate me the movies list oflast 5 movies of this hero, hero name: {hero_name}
# """,
# input_variables=["hero_name"]
# )

# model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# # prompt=template.invoke({
# #     'hero_name':"pawan kalyan"
# # })

# # result=model.invoke(prompt)

# chain= template | model

# result=chain.invoke({
#     'hero_name':"ramcharan"
# })

# print(result.content)

# # st.header("Research Toll")

# # user_input=st.text_input("enter a research paper")

# # if st.button("summarize"):
# #     st.write("prompt")

