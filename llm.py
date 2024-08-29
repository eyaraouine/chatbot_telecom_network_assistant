# Create the LLM
#from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
)

#os.environ["GOOGLE_API_KEY"] =st.secrets["GEMINI_API_KEY"]

#llm = ChatGoogleGenerativeAI(model=st.secrets["GEMINI_MODEL"])