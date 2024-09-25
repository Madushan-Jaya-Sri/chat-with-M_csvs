import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain.chains import create_sql_query_chain
from langchain_openai.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

import re
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import json
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_KEY = os.getenv('OPENAI_KEY')

# Initialize OpenAI LLM
llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0)

# Create a SQLAlchemy engine
engine = create_engine(os.getenv('DB_CONN_STRING'))

# Wrap the engine with SQLDatabase
db = SQLDatabase(engine)

# Create SQL query chain using the SQLDatabase object
write_query = create_sql_query_chain(llm, db )

# Initialize execute_query tool
execute_query = QuerySQLDataBaseTool(db=db, verbose=False)

# Define answer_prompt
answer_prompt = PromptTemplate.from_template(
    """
    Based on the user's question and the SQL result, answer the question either by providing a direct text response or suggesting an appropriate graph type.
    If the SQL Results are not likely to provide the answer to the Question, then re run and get a most suitable SQL Query.
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

    If the answer is a single value or string, provide a direct text answer or
    If the answer needs a graph, provide both visual and text answer.

    Answer format:
    - graph_needed: "yes" or "no"
    - graph_type: one of ['line chart', 'stack bar chart', 'bar chart', 'sankey chart'] (if graph_needed is "yes")
    - data_array: python data list (if graph_needed is "yes")
    - text_answer: The direct answer in point form. use bullets if necessary.(if graph_needed is "no")
    """
)

# Initialize answer parser
answer = answer_prompt | llm | StrOutputParser()

# Create the chain
chain = (RunnablePassthrough.assign(query=write_query)
         .assign(result=itemgetter("query") | execute_query)
         | answer)

# Function to extract fields using regex
def extract_fields(result):
    # Updated regex patterns
    graph_needed_pattern = r'graph_needed:\s*"?(yes|no|[\w\s]+)"?'
    graph_type_pattern = r'graph_type:\s*(\S.*)'
    data_array_pattern = r'\[\s*(.*?)\s*\]'
    text_pattern = r'text_answer:\s*(\S.*)'

    # Extract fields
    graph_needed = re.search(graph_needed_pattern, result, re.IGNORECASE)
    graph_type = re.search(graph_type_pattern, result, re.IGNORECASE)
    data_array = re.search(data_array_pattern, result, re.DOTALL)
    text_output = re.search(text_pattern, result, re.IGNORECASE)

    # Extract and clean values
    graph_needed_value = graph_needed.group(1).strip().lower() if graph_needed else None
    graph_type_value = graph_type.group(1).strip().strip('"') if graph_type else None
    data_array_str = data_array.group(1).strip() if data_array else None
    text_str = text_output.group(1).strip().strip('"') if text_output else None

    # st.write("=========== Data Passed to Plot the Graph =============")
    # st.write(f"Graph Needed: {graph_needed_value}")
    # st.write(f"Graph Type: {graph_type_value}")
    # st.write(f"Data Array: {data_array_str}")
    # st.write("=======================================================")

    if data_array_str:
        # Clean the data array string and convert it to a Python list
        data_string = f"[{data_array_str}]" # Replace single quotes with double quotes
        try:
            # Convert the string to a list of dictionaries
            data_array_value = json.loads(data_string)
        except json.JSONDecodeError:
            st.error("Error decoding JSON from data_array.")
            data_array_value = None
    else:
        data_array_value = None

    return graph_needed_value, graph_type_value, data_array_value, text_str

# Function to plot different types of charts
def plot_chart(graph_needed, graph_type, data_array):
    if graph_needed == "no":
        st.write("No graph needed.")
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
        st.write("Unknown graph type.")

# Function to plot a line chart
def plot_line_chart(data):
    if not data or not isinstance(data, list):
        st.write("Invalid data for line chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        st.write("DataFrame is empty.")
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
    st.pyplot(plt)
    plt.close()

# Function to plot a stacked bar chart
def plot_stack_bar_chart(data):
    if not data or not isinstance(data, list):
        st.write("Invalid data for stacked bar chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        st.write("DataFrame is empty.")
        return

    df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Stacked Bar Chart')
    plt.xlabel(df.columns[0])
    plt.ylabel('Values')
    plt.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# Function to plot a bar chart
def plot_bar_chart(data):
    if not data or not isinstance(data, list):
        st.write("Invalid data for bar chart.")
        return

    df = pd.DataFrame(data)
    if df.empty:
        st.write("DataFrame is empty.")
        return

    df.plot(kind='bar', x=df.columns[0], y=df.columns[1:], figsize=(12, 8))
    plt.title('Bar Chart')
    plt.xlabel(df.columns[0])
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# Function to plot a Sankey chart
def plot_sankey_chart(data):
    if not data or not isinstance(data, list):
        st.write("Invalid data for Sankey chart.")
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
    st.plotly_chart(fig)

def main():
    st.title("SQL Query Assistant with Streamlit")

    # Input area for the SQL query
    query = st.text_area("Enter your SQL question:", value="Give me the all unique investment categories.")

    if st.button("Run Query"):
        if not query.strip():
            st.error("Please enter a valid query.")
            return

        # Define the keyword groups
        keywords1 = ["value1", "value2", "value3","error"]
        keywords2 = ["text_answer"]
        keywords3 = ["yes"]

        output_result = ''
        Text_output = ''

        max_iterations = 5  # To prevent infinite loops
        iteration = 0
        result = chain.invoke({"question": query})

        while iteration < max_iterations:
            # Check if any keyword from keywords1 is in the result
            if any(keyword.lower() in result.lower() for keyword in keywords1):
                st.warning(f"Iteration {iteration+1}: Found keywords1 in result. Re-invoking the chain.")
                result = chain.invoke({"question": query})
                iteration += 1
                continue

             # Check if both keywords2 and keywords3 are in the result (logical AND)
            elif any(keyword.lower() in result.lower() for keyword in keywords2) and any(keyword.lower() in result.lower() for keyword in keywords3):
                output_result = result
                graph_needed_value, graph_type_value, data_array_value, text_str = extract_fields(output_result)
                plot_chart(graph_needed_value, graph_type_value, data_array_value)  # Assuming plot_chart is defined
                st.subheader("Text Output")
                st.write(text_str)
                break

            # Check if any keyword from keywords2 is in the result (only text output)
            elif any(keyword.lower() in result.lower() for keyword in keywords2):
                output_result = result
                graph_needed_value, graph_type_value, data_array_value, text_str = extract_fields(output_result)
                st.subheader("Text Output Only")
                st.write(text_str)
                break

            # If no keywords from any list are found, exit the loop
            else:
                st.info("No relevant keywords found in the result.")
                break
        else:
            st.error("Maximum number of iterations reached. Please check your query or the system.")

if __name__ == "__main__":
    main()
