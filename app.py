import streamlit as st
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import json
import re
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="OpenSearch Natural Language Interface",
    page_icon="🔍",
    layout="centered"
)

# Streamlit app title and description
st.title("OpenSearch Natural Language Interface")
st.markdown("Ask questions in natural language about your OpenSearch data.")

openai_api_key = os.getenv("OPENAI_API_KEY")
opensearch_endpoint = os.getenv("OPENSEARCH_ENDPOINT")
username = os.getenv("OPENSEARCH_USERNAME", "")
password = os.getenv("OPENSEARCH_PASSWORD", "")
port = os.getenv("OPENSEARCH_PORT")  

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Create the model client
model_client = OpenAIChatCompletionClient(
    model='anthropic.claude-3-5-haiku-20241022-v1:0',
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
    model_info={
        "vision": False,
        "family": "unknown",
        "function_calling": True,
        "json_output": True,
    }
)

# Singleton pattern for OpenSearch client
opensearch_client = None

def clean_json_response(response):
    """
    Cleans the response from LLM to ensure it's valid JSON.
    Removes markdown code blocks, comments, and extra whitespace.
    """
    # Remove markdown code blocks if present
    response = re.sub(r'(?:json)?\s*([\s\S]*?)\s*', r'\1', response)
    
    # Remove any JSON comments
    response = re.sub(r'//.*', '', response)
    
    # Look for the first { and last }
    json_start = response.find('{')
    json_end = response.rfind('}')
    
    if json_start != -1 and json_end != -1:
        response = response[json_start:json_end+1]
    
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        return None

def get_opensearch_client():
    global opensearch_client
    
    if opensearch_client is None:
        
        # Initialize OpenSearch client
        opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_endpoint, 'port': port}],
            http_auth=(username, password),
            use_ssl=True,
            verify_certs=False,
            connection_class=RequestsHttpConnection
        )
    
    return opensearch_client

# Function to translate natural language to OpenSearch query using AutoGen
def nl_to_opensearch_query(question, index_name):
    if not openai_api_key:
        st.warning("OpenAI API key not found in environment variables")
        return None
    
    # Get index mapping to understand the data structure
    client = get_opensearch_client()
    mapping = client.indices.get_mapping(index=index_name)
    
    # Extract field names and types
    properties = mapping[index_name]['mappings'].get('properties', {})
    fields_info = {field: prop.get('type', 'unknown') 
                  for field, prop in properties.items()}
    
    # Create AutoGen agent
    agent = AssistantAgent(
        name="QueryTranslator",
        model_client=model_client,  # Use the model_client here, not the OpenSearch client
        system_message=f"""
        You are an AI assistant that translates natural language questions into OpenSearch queries.
        
        The OpenSearch index has the following fields and types:
        {json.dumps(fields_info, indent=2)}
        
        Convert the user's question into a valid OpenSearch query. Focus on creating either:
        1. A match or multi_match query for simple searches
        2. A bool query with must/should/must_not for more complex conditions
        3. Add sort, size, or other parameters as needed
        
        Return ONLY a valid JSON string containing the OpenSearch query body, with no additional text, code fences, or comments.
        
        Rules:
        - Output MUST be a single-line, valid JSON string with no newlines or extra text.
        - Properly escape special characters (e.g., quotes as \", newlines as \\n).
        - Do NOT include any additional text, comments, or formatting outside the JSON string.
        """
    )
    
    # Prompt for the agent
    prompt = f"""
    Convert the following natural language question into a valid OpenSearch query:
    
    Question: {question}
    
    Index fields and types:
    {json.dumps(fields_info, indent=2)}
    
    Return ONLY a valid JSON string containing the OpenSearch query body, with no additional text, code fences, or comments.
    """
    
    # Run the agent
    async def run_agent():
        result = await agent.run(task=prompt)
        return result
    
    try:
        result = asyncio.run(run_agent())
        response = result.messages[1].content
        cleaned_response = clean_json_response(response)
        if cleaned_response:
            return cleaned_response
        else:
            st.error("Failed to parse the generated OpenSearch query")
            return None
    except Exception as e:
        st.error(f"Error generating OpenSearch query: {str(e)}")
        return None

# Function to format OpenSearch results as natural language using AutoGel
def format_results_as_natural_language(results, question):
    if not openai_api_key:
        st.warning("OpenAI API key not found in environment variables")
        return None
    
    # Get the total number of hits
    total_hits = results.get('hits', {}).get('total', {}).get('value', 0)
    hits = results.get('hits', {}).get('hits', [])
    
    # Format the results for the prompt
    formatted_hits = []
    for hit in hits[:10]:  # Limit to first 10 hits for brevity
        source = hit.get('_source', {})
        formatted_hits.append(source)
    
    # Create AutoGen agent
    agent = AssistantAgent(
        name="ResultFormatter",
        model_client=model_client,  # Use the model_client here, not the OpenSearch client
        system_message="""
        You are an AI assistant that summarizes OpenSearch query results into natural language.
        
        Given the user's question and the search results, create a clear, concise summary that:
        1. Answers the question directly
        2. Highlights the most relevant information
        3. Mentions how many results were found in total
        4. Provides specific data points from the results when relevant
        
        Be conversational but informative. If no results were found, suggest possible reasons and alternative queries.
        
        Return ONLY the natural language summary as plain text, with no additional formatting, JSON, or code fences.
        """
    )
    
    # Prompt for the agent
    prompt = f"""
    Summarize the following OpenSearch query results into natural language:
    
    Question: {question}
    
    Total results found: {total_hits}
    
    Search results:
    {json.dumps(formatted_hits, indent=2)}
    
    Return ONLY a plain text summary, with no additional formatting, JSON, or code fences.
    """
    
    # Run the agent
    async def run_agent():
        result = await agent.run(task=prompt)
        return result
    
    try:
        result = asyncio.run(run_agent())
        response = result.messages[1].content
        return response
    except Exception as e:
        st.error(f"Error generating natural language summary: {str(e)}")
        return f"Found {total_hits} results. Error generating summary: {str(e)}"

# Main application logic
def process_question(question, selected_index):
    with st.status("Processing your question...", expanded=True) as status:
        st.write(f"Using index: {selected_index}")
        st.write("Translating to OpenSearch query...")
        query = nl_to_opensearch_query(question, selected_index)
        if query:
            st.write("Query generated:")
            st.code(json.dumps(query, indent=2))
            
            st.write("Executing search...")
            try:
                client = get_opensearch_client()
                results = client.search(
                    body=query,
                    index=selected_index
                )
                
                st.write("Generating natural language response...")
                response = format_results_as_natural_language(results, question)
                status.update(label="Complete!", state="complete")
                return response
                
            except Exception as e:
                st.error(f"Error executing OpenSearch query: {str(e)}")
                status.update(label="Error occurred", state="error")
                return f"Error: {str(e)}"
        else:
            status.update(label="Failed to generate query", state="error")
            return "I couldn't translate your question into a valid OpenSearch query. Please try rephrasing or check your configuration."

# Main interface
st.divider()

if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable is not set")

else:
    try:
        client = get_opensearch_client()
        indices = list(client.indices.get('*').keys())
        
        if indices:
            selected_index = st.selectbox(
                "Select an index to query:", 
                options=indices,
                help="Choose the OpenSearch index you want to query"
            )
            
            question = st.text_input("Ask a question about your data:", 
                                    placeholder="Example: What are the top 5 products by sales?")
            
            if question and selected_index:
                with st.container(border=True):
                    answer = process_question(question, selected_index)
                    st.write("### Answer")
                    st.write(answer)
        else:
            st.error("No indices found in your OpenSearch cluster. Please create at least one index.")
            
    except Exception as e:
        st.error(f"Error connecting to OpenSearch: {str(e)}")
        st.info("Please ensure all required environment variables are set correctly.")

st.divider()