import streamlit as st
from main import sql_agent

st.title("Employee Information: Database Q&A")

question = st.text_input("Question: ")

if question:
    chain = sql_agent()
    response = chain.start_app(question)

    st.header("Answer")
    st.write(response)
    