import streamlit as st


def main():
    title = "Chat with PDFs"
    icon = ":books:"
    st.set_page_config(page_title=title, page_icon=icon)

    st.header(f"{title} {icon}")
    st.text_input("Ask a question about your PDFs:")

    with st.sidebar:
        st.subheader("Your PDFs")
        st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        st.button("Process")

if __name__ == "__main__":
    main()
