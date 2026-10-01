from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm= ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages(
    [("human", "Explain {topic} in 5 simple terms")]
)

chain = prompt | llm | StrOutputParser()
def explain_topic(topic: str) -> str:
    """Explain a given topic in simple terms."""
    return chain.invoke({"topic": topic})

print(explain_topic("docker"))
print(explain_topic("kubernetes"))
print(explain_topic("Terraform"))