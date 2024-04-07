import streamlit as st
from PyPDF2 import PdfReader


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def main():
    title = "Chat with PDFs"
    icon = ":books:"
    st.set_page_config(page_title=title, page_icon=icon)

    st.header(f"{title} {icon}")
    st.text_input("Ask a question about your PDFs:")

    with st.sidebar:
        st.subheader("Your PDFs")
        files = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # fetch file text
                raw_text = get_pdf_text(files)
                # get the text chunks
                # create vector store

if __name__ == "__main__":
    main()
