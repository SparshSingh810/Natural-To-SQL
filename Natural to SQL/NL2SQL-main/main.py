import streamlit as st
from langchain_utils import invoke_chain, get_schema_markdown

st.title("Langchain NL2SQL Chatbot")

with st.sidebar:
    st.header("Database Schema")
    st.caption("Tables and columns available to query. 🔑 = primary key, → = foreign key.")
    try:
        st.markdown(get_schema_markdown())
    except Exception as e:
        st.error(f"Could not load schema: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = invoke_chain(prompt, st.session_state.messages)
            except Exception as e:
                response = f"Something went wrong: {e}"
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
