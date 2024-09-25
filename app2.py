import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.base_language import BaseLanguageModel
from langchain.prompts import BasePromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain.chains.sql_database.query import SQLInput, SQLInputWithTables
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS


import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from typing import Optional, Union, Dict, Any

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
#llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0, model= "gpt-4o")

# Create a SQLAlchemy engine
engine = create_engine(os.getenv('DB_CONN_STRING'))

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

def _strip(text: str) -> str:
    return text.strip()

# Custom function to validate and restrict certain SQL keywords
def validate_sql_query(query: str) -> bool:
    # List of disallowed SQL keywords
    disallowed_keywords = ['SELECT', 'UPDATE', 'DELETE', 'INSERT', 'DROP', 'ALTER']
    
    # Check if any disallowed keyword exists in the query (case insensitive)
    if any(re.search(rf"\b{keyword}\b", query, re.IGNORECASE) for keyword in disallowed_keywords):
        return False
    return True

# Create a custom SQL query chain that validates the query
def create_custom_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 10,
) -> Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]:
    """Create a chain that generates SQL queries with restrictions."""
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT

    if {"input", "top_k", "table_info"}.difference(
        prompt_to_use.input_variables + list(prompt_to_use.partial_variables)
    ):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }

    return (
        RunnablePassthrough.assign(**inputs)
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )

# Additional step to add validation of SQL queries before execution
def execute_safe_sql_query(db, query):
    # Validate the generated query
    if not validate_sql_query(query):
        raise ValueError("Query contains disallowed SQL operations (e.g., SELECT, UPDATE, DELETE).")
    
    # If validation passes, execute the query
    execute_query = QuerySQLDataBaseTool(db=db, verbose=False)
    return execute_query.run(query)

# Define the question and set up the chain
#query = "How Deal Value is divided according to the region and Archetype? "
#query = "Show me the top recipient countries of corperate investments over the last 5 years"
#query = "What is the overall trend of global investments in plastic circularity?"
#query = "⁠For what purposes these investments in plastic circulatory are used for ?"
#query = "⁠what are the archtypes in 2022?"
#query = "show me the way of changing deal value over recent 5 years."
#query = "show me the way of changing deal value over last 5 years by considering exact values."
#query = "how deal value is changed within 2018? "
query = "What was the total spend towards tackling plastic pollution in Indonesia from 2018 to 2023?"
#query = "Give me the all unique investment categories."
#query = "How much of Private Equity money has been received by companies based in Thailand  in 2018?"
#query  = "Which region receives the lowest amount of private investment during the period 2018 to 2023?"
#query = "in which country received the most private investments for biodegradable materials during 2020 to 2022 ."
#query = "What is the total development assistance that has been promised to Africa since 2018? What percentage of the amount committed has been disbursed?"




write_query = create_custom_sql_query_chain(llm, db)

# Create the answer prompt
answer_prompt = PromptTemplate.from_template(
    """
    Based on the user's question and the SQL result, answer the question either by providing a direct explained text response or suggesting an appropriate graph type.
    If the SQL Results are not likely to provide the answer to the question, then re-run and get a more suitable SQL query.
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}

    Please decide if the data should be visualized using one of the following graph types: 'line chart', 'stack bar chart', 'bar chart', 'sankey chart'. 
    If a graph is required, provide the data in the following formats:

    - **Line Chart**: Use a list of dictionaries with x and y values:
      ```python
      [
          {{x-axis name}}: date, {{y-axis name}}: value,
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
          {{category}}: "Category", {{value}}: value,
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

    If the answer for the question is a single value or string, provide a direct explained text answer or
    If the answer needs a graph also, provide both visual and text answer.

    Answer format:
    - graph_needed: "yes" or "no"
    - graph_type: one of ['line chart', 'stack bar chart', 'bar chart', 'sankey chart'] (if graph_needed is "yes")
    - data_array: python data list (if graph_needed is "yes")
    - text_answer: The direct answer (if graph_needed is "no")
    """
)

# Create the tool to execute SQL queries
execute_query_tool = QuerySQLDataBaseTool(db=db, verbose=False)

# Create the answer prompt
answer = answer_prompt | llm | StrOutputParser()

# Combine the chain with answer formatting
chain = (
    RunnablePassthrough.assign(query=write_query)
    .assign(result=itemgetter("query") | execute_query_tool)  # Use execute_query_tool here
    | answer
)
result = chain.invoke({"question": query})
# try :
#     result = chain.invoke({"question": query})
# except BadRequestError as e:
#     print


# print(result)



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

    text_pattern = r'text_answer:\s*(\S.*)'


    text_output = re.search(text_pattern, result)

    text_str = text_output.group(1).strip().strip('"') if text_output else None


    print("=========== data passed to plot the graph =============")
    print(graph_needed_value)
    print(graph_type_value)
    print(data_array_str)
    print("=======================================================")

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

    return graph_needed_value, graph_type_value, data_array_value,text_str



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
    else:

        plt.figure(figsize=(10, 4))
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        plt.title('Data Table')
        plt.show()

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
    else:

        plt.figure(figsize=(10, 4))
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        plt.title('Data Table')
        plt.show() # Create a copy of the DataFrame to avoid modifying the original
        df.set_index(df.columns[0], inplace=True)  # Set the first column as the index
        # Display the DataFrame as a table


        # Plot the stacked bar chart
        df.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Stacked Bar Chart')
        plt.xlabel(df.index.name) 
        plt.ylabel('Values')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
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
    
    else:
        plt.figure(figsize=(10, 4))
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        plt.title('Data Table')
        plt.show()

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
keywords2 = ["text_answer","yes"]
keywords3 = ["text_answer"]
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
        output_result = result
        graph_needed_value, graph_type_value, data_array_value,text_str = extract_fields(output_result)
        print("=========== text output ===========")
        print(text_str)
        plot_chart(graph_needed_value, graph_type_value, data_array_value)
        
        break

    # If no keywords from either list are found, exit the loop
    elif  any(keyword in result for keyword in keywords3):
        output_result = result
        graph_needed_value, graph_type_value, data_array_value,text_str = extract_fields(output_result)
        print("=========== text output only ===========")
        print(text_str)
        break

