import uuid

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001"


st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
)


# Create a conversation ID for this browser session
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())


# Store messages for displaying them in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("RAG Assistant")
st.caption("Ask questions about your documents")


# New chat
if st.button("Clear Chat"):

    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []

    st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
query = st.chat_input("Ask a question...")


if query:

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query,
    })

    with st.chat_message("user"):
        st.write(query)


    # Call FastAPI
    with st.chat_message("assistant"), st.spinner("Thinking..."):

        try:

            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "conversation_id":
                        st.session_state.conversation_id,
                    "query": query,
                },
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            answer = data["answer"]
            sources = data.get("sources", [])
            tools_used = data.get("tools_used", [])

            if tools_used:
                st.markdown("### Tools Used")
                st.write(", ".join(tools_used))

            if sources:
                st.markdown("### Sources")
                for source in sources:
                    document = source.get("source")
                    page = source.get("page")
                    score = source.get("score")
                    if page is not None:
                        st.write(
                            f"{document} — Page {page}"
                        )
                    else:
                        st.write(
                            f"{document}"
                        )

            st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })


        except requests.RequestException as e:

            st.error(
                f"Could not connect to the RAG API: {e}"
            )