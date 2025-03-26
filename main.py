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
pdfs_uploaded=st.file_uploader("Upload your all files here",accept_multiple_files=True)

if(pdfs_uploaded and st.session_state.vectorstore==None):
    with st.spinner("processing"):
        data=''
        if st.button("Process"):
            with st.spinner("Processing..."):
                for pdf in pdfs_uploaded:
                    pdf_reader=PdfReader(pdf)
                    for page in pdf_reader.pages:
                        extracted_text=page.extract_text()
                        data+=extracted_text
        # st.write(f"Extracted Text Length: {len(data)}")
        if not data:
            st.error("No text extracted from the PDFs. Please check the files.")
            st.stop()
        # st.write(data)
        text_splitted=CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        chunks=text_splitted.split_text(data)
        embeddings=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore=FAISS.from_texts(texts=chunks, embedding=embeddings)

if(query and pdfs_uploaded):
    result=''
    with st.spinner("Processing"):
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp-02-05",api_key=GEMINI_API_KEY,temperature=0.7)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=st.session_state.memory,
            chain_type="stuff"
        )
        with st.spinner("Thinking..."):
            response = qa_chain({"question": query})
            answer = response["answer"]
            
        st.subheader("Conversation History")
        # for i, msg in enumerate(st.session_state.memory.memory_key):
        #     if i % 2 == 0:
        #         st.write(f"👤 User: {msg.content}")
        #     else:
        #         st.write(f"🤖 Bot: {msg.content}")
        
        st.write(answer)          
        
    # docs=st.session_state.vectorstore.similarity_search(query)
    # revelent_search="\n".join([x.page_content for x in docs])
    # gemini_prompt="use the following pieces of context to answer the question if you dont know the answer just say Dont make it up And plzz eleborate every answer and make it in how human answer the things."
    # input_prompt=gemini_prompt+"\nContext"+revelent_search+"\nUserQuestion"+query
    
    # result=llm.invoke(input_prompt)