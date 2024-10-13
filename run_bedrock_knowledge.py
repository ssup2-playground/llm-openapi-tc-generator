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


# --- Invoke chain ---
response = chain.invoke("가게 픽업주소안내문구 수정 API를 정보를 알고 싶어")
print(response["response"])

response = chain.invoke("가게 픽업주소안내문구 수정 API의 테스트 케이스를 Python으로 작성해줘")
print(response["response"])
