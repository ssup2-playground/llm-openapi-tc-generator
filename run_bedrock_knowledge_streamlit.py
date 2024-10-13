# --- Init chain with bedrock and knowledge ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever

# Init prompt
template = """
You are a helpful assistant who writes test cases for the API server. Follow these instructions:
1. If you do not know the answer, just say you don't know. Don't make up information.
2. API Spec is included in the following document.
3. Write test cases in accordance with Selenium.
4. When writing a test case, always use 'woowa.in' as the domain, regardless of the actual URL.

Documents: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Init retriever
retriever = AmazonKnowledgeBasesRetriever(
    region_name="us-west-2",
    knowledge_base_id="NVFN52ARPL", # Set knowledge ID
    retrieval_config={
        "vectorSearchConfiguration": {"numberOfResults": 4}
    },
)

# Init model
model = ChatBedrock(
    region_name="us-west-2",
    provider="anthropic",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_kwargs={
        "max_tokens": 8192, # Set model parameters
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    },
)

# Make up chain
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    .assign(response=prompt | model | StrOutputParser())
    .pick(["response", "context"])
)


# --- Streamlit ---
import streamlit as st

# Page title
st.set_page_config(page_title='Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')

# Clear Chat History fuction
def clear_screen():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

with st.sidebar:
    st.title('Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗')
    streaming_on = st.toggle('Streaming')
    st.button('Clear Screen', on_click=clear_screen)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            # Chain Stream
            for chunk in chain.stream(prompt):
                if 'response' in chunk:
                    full_response += chunk['response']
                    placeholder.markdown(full_response)
                else:
                    full_context = chunk
            placeholder.markdown(full_response)
            with st.expander("Show source details >"):
                st.write(full_context)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Chain - Invoke
        with st.chat_message("assistant"):
            response = chain.invoke(prompt)
            st.write(response['response'])
            with st.expander("Show source details >"):
                st.write(response['context'])
            st.session_state.messages.append({"role": "assistant", "content": response['response']})
