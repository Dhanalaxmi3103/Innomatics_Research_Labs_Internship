from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_groq import ChatGroq

from services.transcript_service import extract_transcript
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key safely
os.environ['GROQ_API_KEY'] = os.getenv("groq_api")

llm = ChatGroq(
    model="llama-3.1-8b-instant",   # fast + free
)

# -------- PROMPTS --------
system_message = 'You are a Professional Article Writer.'

human_message = """
Transform transcript into a professional article:

- Ignore intro/promotions
- Focus on technical content
- Use headings, steps, code
- End with summary

{transcript}
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_message),
    HumanMessagePromptTemplate.from_template(human_message)
])

# -------- SHORT --------
base_chain = (
    RunnablePassthrough()
    | RunnableLambda(extract_transcript)
    | prompt
    | llm
    | StrOutputParser()
)

# -------- LONG --------
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    return splitter.split_text(text)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=system_message,
    middleware=[SummarizationMiddleware(model=llm, trigger=("tokens", 1000))]
)

def recursive_summarize(text):
    chunks = get_chunks(text)
    summary = ""

    for chunk in chunks:
        res = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"{summary}\n\n{chunk}"
            }]
        })
        summary = res["messages"][-1].content

    return summary

long_chain = (
    RunnablePassthrough()
    | RunnableLambda(extract_transcript)
    | RunnableLambda(recursive_summarize)
)

# -------- ROUTER --------
def is_long(link):
    return len(extract_transcript(link).split()) > 1000

smart_chain = RunnableBranch(
    (RunnableLambda(is_long), long_chain),
    base_chain
)

def generate_article(url):
    return smart_chain.invoke(url)
