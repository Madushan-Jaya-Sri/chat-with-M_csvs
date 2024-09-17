import os
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

load_dotenv()

# Load OpenAI API key
OPENAI_KEY = os.getenv('OPENAI_KEY')

# Setup the OpenAI LLM
llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0.5)

# Create a SQLAlchemy engine
engine = create_engine('mysql+pymysql://root:root@127.0.0.1:8889/mydatabase')

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)


# Create SQL query chain using the SQLDatabase object
write_query = create_sql_query_chain(llm, db)

execute_query = QuerySQLDataBaseTool(db=db)

chain = write_query | execute_query



from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query and SQL result, answer the user question.
     

    Question :{question}
    SQL query :{query}
    SQL result :{result}
    Answer :

"""
)

answer = answer_prompt | llm |StrOutputParser()

chain = (RunnablePassthrough.assign(query = write_query).assign(result = itemgetter("query") | execute_query) | answer)

print(execute_query)
# Run a query
query = "⁠Show me the top recipient countries of plastic circularity investments over the last 5 years, show me the corresponding years also"
result = chain.invoke({"question": query})

print(result)

