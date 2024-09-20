import os
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool


from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

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


answer_prompt = PromptTemplate.from_template(
    """
    Based on the user's question and the SQL result, answer the question either by providing a direct text response or suggesting an appropriate graph type.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Please decide if the data should be visualized using one of the following graph types: 'line chart', 'stack bar chart', 'bar chart', 'sankey chart'. 
    If a graph is required, provide the data in the following formats:

    - **Line Chart**: Use a list of dictionaries with x and y values:
      ```python
      [
          {{x-axis name}}: date, {{y-axis nam}}e: value,
          ...
      ]
      ```
    - **Stack Bar Chart**: Use a list of dictionaries with categories and stacked values:
      ```python
      [
          {{category}}: "Category", {{value1}}: value1, {{value2}}: value2,
          ...
      ]
      ```
    - **Bar Chart**: Use a list of dictionaries with categories and values:
      ```python
      [
          {{category}}: "Category", {{vlaue}}: value,
          ...
      ]
      ```
    - **Sankey Chart**: Use a list of dictionaries with source, target, and value:
      ```python
      [
          {{source}}: "Source", {{target}}: "Target", {{value}}: value,
          ...
      ]
      ```

    If the answer is a single value or string, provide a direct text answer.

    Answer format:
    - graph_needed: "yes" or "no"
    - graph_type: one of ['line chart', 'stack bar chart', 'bar chart', 'sankey chart'] (if graph_needed is "yes")
    - data_array: python data list (if graph_needed is "yes")
    - text_answer: The direct answer (if graph_needed is "no")
    """
)



answer = answer_prompt | llm |StrOutputParser()

chain = (RunnablePassthrough.assign(query = write_query).assign(result = itemgetter("query") | execute_query) | answer)

print(execute_query)
# Run a query
#query = "What is the overall trend of global investments in plastic circularity?"

query = "How Deal Value is divided according to the region and Archetype? "
result = chain.invoke({"question": query})

print(result)

