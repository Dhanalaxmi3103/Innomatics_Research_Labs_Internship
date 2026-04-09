from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

# Load API key safely
os.environ['GROQ_API_KEY'] = os.getenv("groq_api")
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # fast + free
)

def generate_web_code(article):

    system_message = """You are a Senior Frontend Developer...

--html--
[html]
--html--

--css--
[css]
--css--

--js--
[js]
--js--
"""

    human_message = """
Create production-ready article webpage

CONTENT:
{article_content}
"""

    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message)
    ])

    chain = template | llm | StrOutputParser()

    return chain.invoke({"article_content": article})
