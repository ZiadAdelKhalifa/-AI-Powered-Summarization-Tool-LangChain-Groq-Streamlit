import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_community.document_loaders import SeleniumURLLoader  # New import

## Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

## Get the Groq API Key and URL (YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip():
        st.error("Please enter your Groq API key to proceed.")
    elif not generic_url.strip():
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video URL or website URL")
    else:
        try:
            with st.spinner("Waiting..."):
                ## Initialize the ChatGroq model with a supported model
                llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

                prompt_template = """
                Provide a summary of the following content in 300 words:
                Content: {text}
                """
                prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

                ## Load content based on the URL type
                if "youtube.com" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except Exception:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False)
                        docs = loader.load()
                else:
                    try:
                        # First, try to load using UnstructuredURLLoader (for simple web pages)
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                                                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                        docs = loader.load()
                        if not docs:
                            raise ValueError("Empty content detected, trying Selenium")
                    except Exception:
                        # If UnstructuredURLLoader fails, use SeleniumURLLoader
                        loader = SeleniumURLLoader(urls=[generic_url])
                        docs = loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
