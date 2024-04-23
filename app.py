import streamlit as st

from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from templates import css, bot_template, user_template
from chroma import store_text_in_vector, retriever


def store_pdf_text_to_vector(pdfs):
    for pdf in pdfs:
        file_text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            file_text += page.extract_text()
        store_text_in_vector(file=pdf, raw_text=file_text)

def create_conversation_chain():
    llm = ChatOllama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever(), memory=memory)
    return conversation_chain


def handler_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    title = "Chat with PDFs"
    icon = ":books:"
    st.set_page_config(page_title=title, page_icon=icon)

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = create_conversation_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header(f"{title} {icon}")
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        handler_user_question(user_question)

    with st.sidebar:
        st.subheader("Your PDFs")
        files = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # store file text into chroma db
                store_pdf_text_to_vector(files)

                # create conversation chain
                st.session_state.conversation = create_conversation_chain()

if __name__ == "__main__":
    main()
