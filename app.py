import streamlit as st

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def create_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(chunks):
    embeddings = OllamaEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def create_conversation_chain(vector_store):
    llm = ChatOllama(model="llama2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


def main():
    title = "Chat with PDFs"
    icon = ":books:"
    st.set_page_config(page_title=title, page_icon=icon)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header(f"{title} {icon}")
    st.text_input("Ask a question about your PDFs:")

    with st.sidebar:
        st.subheader("Your PDFs")
        files = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # fetch file text
                raw_text = get_pdf_text(files)

                # create text chunks
                chunks = create_text_chunks(raw_text)

                # create vector store
                vector_store = create_vector_store(chunks)

                # create conversation chain
                st.session_state.conversation = create_conversation_chain(vector_store)

if __name__ == "__main__":
    main()
