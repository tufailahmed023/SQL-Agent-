import streamlit as st
from main import sql_agent


st.title("Employee DATABASE Q&A")

st.image("https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy.gif", width=250)


question = st.text_input("Ask me anything about employee information !")

if st.button("Ask"):
    if question:
        chain = sql_agent()
        response = chain.start_app(question)

        # Displaying the answer
        st.header("Answer")
        st.write(response)
    else:
        st.warning("Please enter a question first!")



