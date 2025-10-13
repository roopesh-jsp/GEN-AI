import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_textcontent(pdfs):
    raw_text=""
    for pdf in pdfs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text+=page.extract_text()
    return raw_text
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def main():
    load_dotenv()

    st.set_page_config(page_title="chat with PDF's",page_icon=":books:")

    st.header("chat with your PDF's and get the insights")
    st.text_input("ask questions related to pdfs ")

    with st.sidebar:
        st.header("Your PDFs")
        pdfs=st.file_uploader("upload your PDFs and click on process button below",accept_multiple_files=True)
        if st.button("process"):
            with st.spinner("processing"):
                # getting the pdf text 
                raw_text=get_pdf_textcontent(pdfs)
                
                # extract the chunks in pdf 
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)

                #create embeddings
                
                # create a vetor store and store the embeddings there 

if __name__=='__main__':
    main()