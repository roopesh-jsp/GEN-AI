import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_huggingface import HuggingFaceEmbeddings

st.markdown("""
    <style>
    /* Make the input and button align on the same height */
    div[data-testid="stHorizontalBlock"] div[data-testid="stVerticalBlock"] {
        display: flex;
        align-items: center;
    }

    # /* Optional: make the input box and button match in height */
    # .stTextInput > div > div > input {
    #     height: 3rem;
    #     font-size: 1rem;
    #     padding:10px;
    # }
     
    # .stButton > button {
    #     height: 3rem;
    #     border-radius: 0.5rem;
    # }
    </style>
""", unsafe_allow_html=True)



def get_pdf_textcontent(pdfs):
    raw_text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    # Custom prompt template for better responses
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise and relevant to the question.

Context: {context}

Question: {question}

Helpful Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("⚠️ Please upload and process PDFs first!")
        return
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Invoke the chain with the question
            response = st.session_state.conversation.invoke({"query": user_question})
            
            # Store the Q&A in session state
            st.session_state.messages.append({
                "question": user_question,
                "answer": response['result']
            })
            
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
            import traceback
            with st.expander("Show full error"):
                st.code(traceback.format_exc())
            return


def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚",
        layout="wide"
    )
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found in .env file!")
        st.stop()
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.title("📚 Chat with Your PDFs")
    st.markdown("Upload your PDF documents and ask questions about their content!")
    
    # Main area - Question input and responses
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input(
            "💬 Ask a question about your documents:",
            placeholder="e.g., What is the main topic of this document?",
            key="user_input",
            label_visibility="collapsed"
        )

    with col2:
        if st.button("🔍 Ask", use_container_width=True, disabled=st.session_state.conversation is None):
            if user_question:
                handle_user_input(user_question)

    # Display conversation history
    if st.session_state.messages:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for i, msg in enumerate(reversed(st.session_state.messages)):
            with st.container():
                st.markdown(f"**🧑 Question {len(st.session_state.messages) - i}:**")
                st.info(msg['question'])
                st.markdown(f"**🤖 Answer:**")
                st.success(msg['answer'])
                st.markdown("---")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Document Management")
        
        pdfs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files"
        )
        
        if st.button("🚀 Process Documents", use_container_width=True):
            if not pdfs:
                st.error("❌ Please upload at least one PDF file!")
            else:
                with st.spinner("⏳ Processing your documents..."):
                    try:
                        # Get PDF text
                        raw_text = get_pdf_textcontent(pdfs)
                        
                        if not raw_text.strip():
                            st.error("❌ No text could be extracted from the PDFs!")
                            return
                        
                        st.info(f"📄 Extracted {len(raw_text):,} characters")
                        
                        # Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.info(f"✂️ Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        with st.spinner("🔮 Creating vector embeddings..."):
                            vectorstore = get_vectorstore(text_chunks)
                        st.success("✅ Vector store created!")
                        
                        # Create conversation chain
                        with st.spinner("🤖 Initializing AI model..."):
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                        
                        # Reset conversation history
                        st.session_state.messages = []
                        
                        st.success("✅ Ready to chat! Ask your questions above.")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")
                        import traceback
                        with st.expander("Show full error"):
                            st.code(traceback.format_exc())
        
        # Clear conversation button
        if st.session_state.messages:
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Display info
        st.markdown("---")
        st.markdown("### ℹ️ How to use:")
        st.markdown("""
        1. **Upload** one or more PDF files
        2. Click **Process Documents**
        3. **Ask questions** in the text box
        4. Get **AI-powered answers**!
        """)
        
        # Status section
        st.markdown("---")
        st.markdown("### 📊 Status")
        
        if st.session_state.conversation:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ No documents loaded")
            
        if pdfs:
            st.info(f"📚 {len(pdfs)} file(s) selected")
            
        if st.session_state.messages:
            st.info(f"💬 {len(st.session_state.messages)} Q&A pairs")
        
        # Model info
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.caption("Using: Google gemini-2.5-flash")
        st.caption("Embeddings: all-MiniLM-L6-v2")


if __name__ == '__main__':
    main()