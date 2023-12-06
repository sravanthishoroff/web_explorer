import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

import os

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"  # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "YOUR_CSE_ID"  # Get it at https://programmablesearchengine.google.com/
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
os.environ["OPENAI_API_KEY"] = ""  # Get it at https://beta.openai.com/account/api-keys

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():
    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm,
        search=search,
        num_search_results=3
    )

    return web_retriever, llm

def stream_handler(container, initial_text=""):
    text = initial_text

    def on_llm_new_token(token: str, **kwargs) -> None:
        nonlocal text
        text += token
        container.info(text)

    return on_llm_new_token

def print_retrieval_handler(container):
    retrieval_container = container.expander("Context Retrieval")

    def on_retriever_start(query: str, **kwargs):
        retrieval_container.write(f"**Question:** {query}")

    def on_retriever_end(documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            retrieval_container.write(f"**Results from {source}**")
            retrieval_container.text(doc.page_content)

    return on_retriever_start, on_retriever_end

st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
        "I can be configured to use different modes: public API or private (no data sharing).`")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings(), None
 
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input
question = st.text_input("`Ask a question:`")

if question:
    # Generate answer (w/ citations)
    import logging

    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_start_cb, retrieval_end_cb = print_retrieval_handler(st.container())
    answer = st.empty()
    stream_handler_cb = stream_handler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question}, callbacks=[retrieval_start_cb, retrieval_end_cb, stream_handler_cb])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
