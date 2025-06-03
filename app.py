import streamlit as st
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import json
import re
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
from desc_find import example_usage
from fetch_rules_wazuh import fetch_wazuh_rules_descriptions

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

def load_rule_descriptions(question):
    """
    Load and extract descriptions from the JSON file.
    Returns a list of unique descriptions.
    """
    json_file_path="wazuh_rules.json"
    try:
        # with open(json_file_path, 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        
        # descriptions = []
        # for rule in data:
        #     description = rule.get('description', '').strip()
        #     if description and description not in descriptions:
        #         descriptions.append(description)
        descriptions = fetch_wazuh_rules_descriptions()
        final_descriptions = example_usage(descriptions, question)
        return final_descriptions
    except Exception as e:
        st.error(f"Error loading rule descriptions: {str(e)}")
        return []

def find_relevant_rule_description(question, descriptions):
    """
    Use LLM to identify if the question is related to rules/descriptions
    and return the most relevant description.
    """
    if not descriptions:
        return None
    
    # Create AutoGen agent for rule identification
    agent = AssistantAgent(
        name="RuleIdentifier",
        model_client=model_client,
        system_message=f"""
        You are an AI assistant that identifies if a user question specifically needs rule descriptions for searching.
        
        Available rule descriptions:
        {json.dumps(descriptions, indent=2)}
        
        IMPORTANT: Only return a rule description if the user is asking about:
        1. Rule content/behavior (what does a rule do, how does it work)
        3. Rule templates or generic rule information
        4. Security rule explanations or descriptions
        
        DO NOT return a rule description if the user is asking about:
        1. Specific rule IDs (e.g., "rule ID", "rule with ID", "id equals")
        2. Rule levels, status, or other metadata fields
        3. Specific numerical or field-based queries
        4. Alerts, logs, or events with specific rule identifiers
        
        Return a JSON response with the following format:
        {{
            "is_rule_related": true/false,
            "relevant_description": "exact description from the list or null",
        }}
        
        
        Return ONLY valid JSON, no additional text or formatting.
        """
    )
    
    prompt = f"""
    Analyze this user question and determine if it specifically needs rule descriptions for searching:
    
    Question: {question}
    
    Consider:
    - Does this ask about rule content, behavior, or categories?
    - Or does this ask about specific rule IDs, levels, or metadata fields?
    - Questions about specific IDs, numbers, or field values should NOT use descriptions.
    
    Return JSON response indicating if rule descriptions are needed and which one.
    """
    
    async def run_agent():
        result = await agent.run(task=prompt)
        return result
    
    try:
        result = asyncio.run(run_agent())
        response = result.messages[1].content
        cleaned_response = clean_json_response(response)
        rule_info = json.loads(cleaned_response)
        
        if rule_info.get('is_rule_related', False):
            return rule_info.get('relevant_description')
        return None
        
    except Exception as e:
        st.error(f"Error identifying rule relevance: {str(e)}")
        return None

def extract_fields_info(properties, parent_prefix=""):
    """Recursively extract field names and types from OpenSearch mappings"""
    fields_info = {}

    for field, prop in properties.items():
        full_path = f"{parent_prefix}{field}"

        # Check if this is a nested object with its own properties
        if "properties" in prop:
            # Add this field as an object type
            fields_info[full_path] = "object"
            # Add all nested fields with proper path
            nested_fields = extract_fields_info(prop["properties"], f"{full_path}.")
            fields_info.update(nested_fields)
        else:
            # It's a primitive field
            field_type = prop.get("type", "unknown")
            fields_info[full_path] = field_type

    return fields_info

def clean_json_response(response):
    """
    Cleans the response from LLM to ensure it's valid JSON.
    Attempts to fix common issues like missing closing braces.
    """
    try:
        # Remove markdown code block markers
        response = re.sub(r'^```(?:json)?\s*\n?|\n?```$', '', response.strip(), flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove any comments
        response = re.sub(r'//.*$', '', response, flags=re.MULTILINE)
        
        # Strip whitespace
        response = response.strip()
        
        # Try to parse the response as-is
        try:
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # Attempt to fix missing closing braces
            brace_count = response.count('{') - response.count('}')
            if brace_count > 0:
                # Add missing closing braces
                response += '}' * brace_count
                try:
                    json.loads(response)
                    print(f"Fixed JSON by adding {brace_count} closing brace(s)")
                    return response
                except json.JSONDecodeError as e:
                    print(f"Failed to fix JSON: {e}")
                    print(f"Repaired response: {repr(response)}")
            
            # If parsing still fails, extract the JSON object
            start = response.find('{')
            if start == -1:
                raise ValueError("No JSON object found")
                
            # Find the last valid closing brace
            brace_count = 0
            end = -1
            for i in range(start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            
            if end == -1:
                # Try to append closing braces to make it valid
                response = response[:start] + response[start:] + '}' * brace_count
                try:
                    json.loads(response)
                    print(f"Fixed JSON by appending {brace_count} closing brace(s)")
                    return response
                except json.JSONDecodeError as e:
                    raise ValueError(f"Unable to fix JSON: {e}")
            
            json_str = response[start:end + 1]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                raise ValueError(f"Extracted JSON is invalid: {e}")
                
    except Exception as e:
        print(f"JSON cleaning error: {e}")
        print(f"Raw response: {repr(response)}")
        raise ValueError(f"Failed to clean JSON response: {e}")
    

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
def nl_to_opensearch_query(question, index_name, rule_description=None):
    if not openai_api_key:
        st.warning("OpenAI API key not found in environment variables")
        return None
    
    # Get index mapping to understand the data structure
    client = get_opensearch_client()
    mapping = client.indices.get_mapping(index=index_name)
    
    # Extract field names and types
    properties = mapping[index_name]['mappings'].get('properties', {})
    fields_info = extract_fields_info(properties)

    # Enhanced system message with rule description context
    rule_context = ""
    if rule_description:
        rule_context = f"""
        
        IMPORTANT: The user's question is related to this specific rule description:
        "{rule_description}"
        
        When constructing the query, use this rule description in your search criteria.
        If there's a 'description' field in the index, include a match query for this specific description.
        """
    
    # Create AutoGen agent
    agent = AssistantAgent(
        name="QueryTranslator",
        model_client=model_client,  # Use the model_client here, not the OpenSearch client
        system_message=f"""
        You are an AI assistant that translates natural language questions into OpenSearch queries.
        
        The OpenSearch index has the following fields and types:
        {fields_info}
        
        Convert the user's question into a Valid OpenSearch query. Focus on creating either:
        1. A match or multi_match query for simple searches.
        2. A bool query with must/should/must_not for more complex conditions.
        3. Only include sort, size if explicitly specified in the input otherwise do not mention it.
        
        Return ONLY a valid JSON string containing the OpenSearch query body, with no additional text, code fences, or comments.
        
        Rules:
        - Output MUST be a single-line, VALID JSON string with no newlines or extra text.
        - Always include the "query" wrapper object.
        - Properly escape special characters (e.g., quotes as \", newlines as \\n).
        - Do NOT include any additional text, comments, or formatting outside the JSON string.
        """
    )
    
    # Enhanced prompt with rule description context
    rule_prompt_context = ""
    if rule_description:
        rule_prompt_context = f"""
        
        Context: This question is related to the rule description: "{rule_description}"
        Please incorporate this rule description into your OpenSearch query construction.
        """
    
    # Prompt for the agent
    prompt = f"""
    Convert the following natural language question into a Valid OpenSearch query:
    
    Question: {question}
    {rule_prompt_context}
    
    Index fields and types:
    {json.dumps(fields_info, indent=2)}
    
    Return ONLY a VALID JSON string containing the OpenSearch query body, with no additional text, code fences, or comments.
    """
    
    # Run the agent
    async def run_agent():
        result = await agent.run(task=prompt)
        # print("res: ",result)
        return result
    
    try:
        result = asyncio.run(run_agent())
        response = result.messages[1].content
        print("ressponse: ",response)
        cleaned_response = clean_json_response(response)
        print("cleaned response",cleaned_response)
        if cleaned_response:
            print("json.loads(cleaned_response): ",json.loads(cleaned_response))
            return json.loads(cleaned_response)

        else:
            st.error("Failed to parse the generated OpenSearch query")
            return None
    except Exception as e:
        st.error(f"Error generating OpenSearch query: {str(e)}")
        return None

# Function to format OpenSearch results as natural language using AutoGen
def format_results_as_natural_language(results, question, rule_description=None):
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
    
    # Rule context for response formatting
    rule_context = ""
    if rule_description:
        rule_context = f"""
        
        Note: This query was related to the rule description: "{rule_description}"
        Please mention this context in your summary when relevant.
        """
    
    # Create AutoGen agent
    agent = AssistantAgent(
        name="ResultFormatter",
        model_client=model_client,
        system_message=f"""
        You are an AI assistant that summarizes OpenSearch query results into natural language.
        
        Given the user's question and the search results, create a clear, concise summary that:
        1. Answers the question directly
        2. Highlights the most relevant information
        3. Mentions how many results were found in total
        4. Provides specific data points from the results when relevant
        5. If rule description context is provided, mention it appropriately
        
        Be conversational but informative. If no results were found, suggest possible reasons and alternative queries.
        
        Return ONLY the natural language summary as plain text, with no additional formatting, JSON, or code fences.
        """
    )
    
    # Prompt for the agent
    prompt = f"""
    Summarize the following OpenSearch query results into natural language:
    
    Question: {question}
    {rule_context}
    
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
        
        # Load rule descriptions and check if question is rule-related
        st.write("Checking if question is related to rules...")
        descriptions = load_rule_descriptions(question)
        relevant_rule_description = find_relevant_rule_description(question, descriptions)
        
        if relevant_rule_description:
            st.write(f"✅ Rule-related question detected!")
            st.write(f"Relevant rule: {relevant_rule_description}")
        else:
            st.write("ℹ️ General question (not rule-specific)")
        
        st.write("Translating to OpenSearch query...")
        query = nl_to_opensearch_query(question, selected_index, relevant_rule_description)
        print("query: ",query)
        if query:
            st.write("Query generated:")
            st.code(json.dumps(query, indent=2))
            
            # st.write("Executing search...")
            # try:
            #     client = get_opensearch_client()
            #     results = client.search(
            #         body=query,
            #         index=selected_index
            #     )
                
            #     st.write("Generating natural language response...")
            #     response = format_results_as_natural_language(results, question, relevant_rule_description)
            #     status.update(label="Complete!", state="complete")
            #     return response
                
            # except Exception as e:
            #     st.error(f"Error executing OpenSearch query: {str(e)}")
            #     status.update(label="Error occurred", state="error")
            #     return f"Error: {str(e)}"
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
                                    placeholder="Example: What are the top 5 products by sales? or Show me firewall rules")
            
            if question and selected_index:
                with st.container(border=True):
                    answer = process_question(question, selected_index)
                    # st.write("### Answer")
                    # st.write(answer)
        else:
            st.error("No indices found in your OpenSearch cluster. Please create at least one index.")
            
    except Exception as e:
        st.error(f"Error connecting to OpenSearch: {str(e)}")
        st.info("Please ensure all required environment variables are set correctly.")

st.divider()