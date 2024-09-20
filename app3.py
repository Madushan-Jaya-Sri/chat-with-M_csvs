import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
import plotly.graph_objects as go
import re
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import os
import pandas as pd

load_dotenv()

# Load OpenAI API key
OPENAI_KEY = os.getenv('OPENAI_KEY')

# Setup the OpenAI LLM
llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0)

# Create a SQLAlchemy engine
engine = create_engine(os.getenv('DB_CONN_STRING'))

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

# precautions_text = """
# - Avoid using SELECT * unless absolutely necessary.
# - Ensure the query is optimized by using proper indexes if applicable.
# - Make sure to filter sensitive data such as passwords or personal identifiable information (PII).
# - Do not write queries that modify the data (e.g., INSERT, UPDATE, DELETE).
# """
# Run a query
#query = "What is the overall trend of global investments in plastic circularity?"

#query = "How Deal Value is divided according to the region and Archetype? "
query = "Show me the top recipient countries of corperate investments over the last 5 years"
#query = "What is the overall trend of global investments in plastic circularity?"
#query = "⁠For what purposes these investments in plastic circulatory are used for ?"
#query = "⁠what are the archtypes in 2022?"
#query = "show me the way of changing deal value over recent 5 years."
#query = "show me the way of changing deal value over recent 5 years by considering exact values."
#query = "how deal value is changed within 2018? "
# Create a custom prompt template incorporating the precautions
#top_k=20


# SQL_custom_prompt_template = PromptTemplate.from_template(
    
    
    
#     """
#     You are an expert SQL query writer. 
#     Follow these precautions while writing SQL queries:
#     {precautions_text}
#     In case of asking a question that will returns large number of output rows, please limit the results for {top_k} only.
#     Write an optimized SQL query to answer the following question:
#     {input}
#     """
    
# )

# Create SQL query chain using the SQLDatabase object
#write_query = create_sql_query_chain(llm, db ,prompt=SQL_custom_prompt_template)
write_query = create_sql_query_chain(llm, db )
print(write_query)

#print(write_query)

execute_query = QuerySQLDataBaseTool(db=db,verbose= False)



answer_prompt = PromptTemplate.from_template(
    """
    Based on the user's question and the SQL result, answer the question either by providing a direct text response or suggesting an appropriate graph type.
    If the SQL Reuslts are not likely to provide the answer to the Question, then re run and get a most suitable SQL Query.
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


result = chain.invoke({"question": query})



# Function to extract fields using regex
def extract_fields(result):
    # Updated regex patterns
    graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
    graph_type_pattern = r'graph_type:\s*(\S.*)'
    data_array_pattern = r'\[\s*(.*?)\s*\]'

    # Extract fields
    graph_needed = re.search(graph_needed_pattern, result)
    graph_type = re.search(graph_type_pattern, result)
    data_array = re.search(data_array_pattern, result, re.DOTALL)

    # Extract and clean values
    graph_needed_value = graph_needed.group(1) if graph_needed else None
    graph_type_value = graph_type.group(1).strip().strip('"') if graph_type else None
    data_array_str = data_array.group(1) if data_array else None

    print("=========== data passed to plot the graph =============")
    print(graph_needed_value)
    print(graph_type_value)
    print(data_array_str)

    if data_array_str:
        # Clean the data array string and convert it to a Python list
        data_string = f"[{data_array_str}]" # Replace single quotes with double quotes
        try:
            # Convert the string to a list of dictionaries
            data_array_value = json.loads(data_string)
          # Convert string to Python list
        except json.JSONDecodeError:
            print("Error decoding JSON from data_array.")
            data_array_value = None
    else:
        data_array_value = None

    return graph_needed_value, graph_type_value, data_array_value

# Function to plot different types of charts
def plot_chart(graph_needed, graph_type, data_array):
    if graph_needed == "no":
        print("No graph needed.")
        return

    if graph_type == "line chart":
        plot_line_chart(data_array)
    elif graph_type == "stack bar chart":
        plot_stack_bar_chart(data_array)
    elif graph_type == "bar chart":
        plot_bar_chart(data_array)
    elif graph_type == "sankey chart":
        plot_sankey_chart(data_array)
    else:
        print("Unknown graph type.")

# Function to plot a line chart
def plot_line_chart(data):
    if not data or not isinstance(data, list):
        print("Invalid data for line chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        print("DataFrame is empty.")
        return

    x_col = df.columns[0]
    plt.figure(figsize=(10, 6))
    for column in df.columns[1:]:
        plt.plot(df[x_col], df[column], marker='o', label=column)
    plt.title('Line Chart')
    plt.xlabel(x_col)
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot a stacked bar chart
def plot_stack_bar_chart(data):
    if not data or not isinstance(data, list):
        print("Invalid data for stacked bar chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        print("DataFrame is empty.")
        return

    df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Stacked Bar Chart')
    plt.xlabel(df.columns[0])
    plt.ylabel('Values')
    plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Function to plot a bar chart
def plot_bar_chart(data):
    if not data or not isinstance(data, list):
        print("Invalid data for bar chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        print("DataFrame is empty.")
        return

    df.plot(kind='bar', x=df.columns[0], y=df.columns[1:], figsize=(12, 8))
    plt.title('Bar Chart')
    plt.xlabel(df.columns[0])
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Function to plot a Sankey chart
def plot_sankey_chart(data):
    if not data or not isinstance(data, list):
        print("Invalid data for Sankey chart.")
        return

    sources = [d.get('source') for d in data]
    targets = [d.get('target') for d in data]
    values = [d.get('value') for d in data]

    unique_nodes = list(set(sources + targets))
    node_indices = {node: idx for idx, node in enumerate(unique_nodes)}

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=unique_nodes
        ),
        link=dict(
            source=[node_indices.get(src) for src in sources],
            target=[node_indices.get(tgt) for tgt in targets],
            value=values
        )
    ))

    fig.update_layout(title_text='Sankey Diagram', font_size=10)
    fig.show()








keywords1 = ["value1", "value2", "value3","error"]
keywords2 = ["text_answer"]
output_result = ''
Text_output = ''

while True:
    # Check if any keyword from keywords1 is in the result
    if any(keyword in result for keyword in keywords1):
        print(result)
        result = chain.invoke({"question": query})

    # Check if any keyword from keywords2 is in the result
    elif any(keyword in result for keyword in keywords2):
        Text_output = result
        print(Text_output)
        break

    # If no keywords from either list are found, exit the loop
    else:
        output_result = result
        graph_needed_value, graph_type_value, data_array_value = extract_fields(output_result)
        plot_chart(graph_needed_value, graph_type_value, data_array_value)
        break

