import streamlit as st
from langchain_openai import ChatOpenAI

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    partial_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        llm = ChatOpenAI(model="gpt-4.1", temperature=0, streaming=True)
        for chunk in llm.stream(st.session_state.history[-30:]):
            partial_text += chunk.content or ""  # type: ignore
            placeholder.markdown(partial_text)

    st.session_state.history.append({"role": "assistant", "content": partial_text})
