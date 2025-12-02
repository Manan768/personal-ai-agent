import streamlit as st
from agent import ask

st.set_page_config(page_title="Personal AI Agent", layout="centered")

st.title("🤖 Personal AI Agent")

# Chat messages stored in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# User input
user_input = st.chat_input("Ask anything...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Call the agent
        response = ask(user_input)

        # Display assistant response
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"Error: {e}")
