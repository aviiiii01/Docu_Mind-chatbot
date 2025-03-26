from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from pypdf import PdfReader
import streamlit as st
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

load_dotenv() 

st.set_page_config("Botty mera bhai!")
st.header("Chat with Multiple PDFs:books:")
query=st.text_input("Ask your question about the docs!....")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
db=None
with st.sidebar:
    st.subheader("Browse your document here Bro!..")
    pdfs_uploaded = st.file_uploader("Upload your all files here", accept_multiple_files=True)
    
    btn = st.button("Process", key="process_btn")  # Ensure unique key
    
    if btn and pdfs_uploaded and st.session_state.vectorstore is None:
        with st.spinner("Processing..."):  # Spinner starts before processing
            data = ''
            for pdf in pdfs_uploaded:
                pdf_reader = PdfReader(pdf)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    data += extracted_text
            
            # Ensure extracted text exists
            if not data:
                st.error("No text extracted from the PDFs. Please check the files.")
                st.stop()

            # Split text into chunks
            text_splitted = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )
            chunks = text_splitted.split_text(data)

            # Create embeddings and store in FAISS
            embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

            st.success("✅ Documents processed successfully!")
            
if(query and pdfs_uploaded):
    result=''
    with st.spinner("Processing"):
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",api_key=GEMINI_API_KEY,temperature=0.7)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),#Converts the vector store into a retriever -> A Retriever is a standardized interface for fetching documents, which can later be used with LangChain's retrieval-based chains.
            memory=st.session_state.memory,
            chain_type="stuff"
        )
    with st.spinner("Thinking..."):
        response = qa_chain({"question": query})
        answer = response["answer"]
        
    if(st.session_state.memory==None):
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background-color: #E8F5E9;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 5px 0;
                    text-align: left;
                    font-weight: bold;
                    color: #1B5E20;
                ">
                    👤 <strong>User:</strong> {query}
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown(
                    f"""
                    <div style="
                        background-color: #E8F5E9;
                        padding: 10px;
                        border-radius: 10px;
                        margin: 5px 0;
                        text-align: left;
                        font-weight: bold;
                        color: #1B5E20;
                    ">
                        🤖 <strong>User:</strong> {answer}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    else:
        if "memory" in st.session_state and st.session_state.memory.chat_memory:
            chat_messages = st.session_state.memory.chat_memory.messages[::-1]  # Reverse order

            for i in range(0, len(chat_messages), 2):  # Process in pairs (User -> Bot)
                with st.container():
                    if i+1 < len(chat_messages):  # User message
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #E8F5E9;
                                padding: 10px;
                                border-radius: 10px;
                                margin: 5px 0;
                                text-align: left;
                                font-weight: bold;
                                color: #1B5E20;
                            ">
                                🤖 <strong>User:</strong> {chat_messages[i+1].content}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    if i < len(chat_messages):  # Bot response
                        st.markdown(
                            f"""
                            <div style="
                                background-color: #E3F2FD;
                                padding: 10px;
                                border-radius: 10px;
                                margin: 5px 0;
                                text-align: left;
                                font-weight: bold;
                                color: #0D47A1;
                            ">
                                👤 <strong>Bot:</strong> {chat_messages[i].content}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
        
        #st.write(answer)          
        
    # docs=st.session_state.vectorstore.similarity_search(query)  #-> Directly searches for similar documents
    # revelent_search="\n".join([x.page_content for x in docs])
    # gemini_prompt="use the following pieces of context to answer the question if you dont know the answer just say Dont make it up And plzz eleborate every answer and make it in how human answer the things."
    # input_prompt=gemini_prompt+"\nContext"+revelent_search+"\nUserQuestion"+query
    # llm=ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",api_key=GEMINI_API_KEY,temperature=0.7)

    # result=llm.invoke(input_prompt)