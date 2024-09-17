import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
OPENAI_KEY = os.environ['OPENAI_KEY']

# Initialize the language model
llm = ChatOpenAI(api_key=OPENAI_KEY, temperature=0.5)

# Streamlit app title
st.title("CSV File Processor with LangChain")

# File uploader for multiple CSV files
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# Initialize a session state to store the agents and dataframes
if "agents" not in st.session_state:
    st.session_state.agents = {}
    st.session_state.dataframes = {}

# Processing the uploaded files and creating agents
if uploaded_files:
    st.success("Files uploaded successfully! Now you can ask your questions.")
    for uploaded_file in uploaded_files:
        # Load the uploaded CSV into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Save the CSV file to the current directory
        csv_file = uploaded_file.name
        df.to_csv(csv_file, index=False)
        
        # Create an agent for the CSV file and store it in session state
        agent_executer = create_csv_agent(llm, csv_file, verbose=True, allow_dangerous_code=True)
        st.session_state.agents[csv_file] = agent_executer
        st.session_state.dataframes[csv_file] = df

# Input area to ask questions once files are uploaded and processed
if st.session_state.agents:
    selected_file = st.selectbox("Select a file to query", list(st.session_state.agents.keys()))
    question = st.text_input("Ask a question about the selected file")
    
    if question:
        # Get the corresponding agent and DataFrame for the selected file
        agent_executer = st.session_state.agents[selected_file]
        df = st.session_state.dataframes[selected_file]
        
        # Detect if the question is about generating a pie chart
        if "pie chart" in question.lower():
            # Assume the question specifies the column name for the pie chart
            column_name = question.split("pie chart for")[1].strip()
            
            if column_name in df.columns:
                # Generate and display the pie chart
                fig, ax = plt.subplots()
                df[column_name].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_ylabel('')  # Hide the y-label
                ax.set_title(f"Pie Chart for {column_name}")
                st.pyplot(fig)
            else:
                st.error(f"Column '{column_name}' not found in {selected_file}")
        else:
            # Invoke the agent with the user-provided question
            response = agent_executer.invoke(question)
            
            # Extract the input and output from the response
            input_text = response.get('input', '')
            output_text = response.get('output', '')
            
            # Display the results in a nicely formatted way
            st.markdown(f"### Response for **{selected_file}**")
            st.markdown(f"**Question:** `{input_text}`")
            st.markdown(f"**Answer:** `{output_text}`")
