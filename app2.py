import streamlit as st
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import json
import re
from opensearchpy import OpenSearch, RequestsHttpConnection
from dotenv import load_dotenv
from desc_find import example_usage, OpenAISemanticMatcher
from fetch_rules_wazuh import fetch_wazuh_rules_descriptions


load_dotenv()
matcher = OpenAISemanticMatcher()

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
    
def load_rule_descriptions_v2():
    """
    Load and extract descriptions from the JSON file.
    Returns a list of unique descriptions.
    """
    json_file_path="wazuh_rules.json"
    try:
        descriptions = fetch_wazuh_rules_descriptions()
        embeddings = matcher.create_store_descriptions(descriptions)
        print("MEBDDINGS: ", embeddings)
        if len(embeddings) > 0:
            return True
        
        return False
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

def analyze_fields_for_query(fields_info, question):
    """
    Analyze which fields are most relevant for the given question.
    Returns categorized field information for better query construction.
    """
    agent = AssistantAgent(
        name="FieldAnalyzer",
        model_client=model_client,
        system_message=f"""
        You are an expert at analyzing database schema and determining which fields are most relevant for search queries.
        
        Available fields and their types:
        {json.dumps(fields_info, indent=2)}
        
        Categorize the fields into:
        1. TEXT_FIELDS: Fields that contain searchable text content (type: text, keyword with text-like names)
        2. FILTER_FIELDS: Fields good for filtering (dates, numbers, exact matches, IDs)
        3. NESTED_FIELDS: Complex nested objects that might need special handling
        4. RELEVANT_FIELDS: Fields that seem most relevant to the user's question
        
        Return a JSON response with field categorization and reasoning.
        """
    )
    
    prompt = f"""
    Analyze these fields for the question: "{question}"
    
    Categorize the fields and identify:
    1. Which fields likely contain searchable text content
    2. Which fields are good for filtering/exact matching  
    3. Which fields are most relevant to this specific question
    4. Any nested fields that need special handling
    
    Return JSON with your analysis and reasoning.
    """
    
    async def run_analysis():
        result = await agent.run(task=prompt)
        return result
    
    try:
        result = asyncio.run(run_analysis())
        response = result.messages[1].content
        cleaned_response = clean_json_response(response)
        return json.loads(cleaned_response)
    except Exception as e:
        st.error(f"Error analyzing fields: {str(e)}")
        return None

# Function to translate natural language to OpenSearch query using AutoGen
def deep_research_query_generation(question, index_name, rule_description=None, max_iterations=3):
    """
    Enhanced query generation with iterative refinement like ChatGPT Deep Research
    """
    if not openai_api_key:
        st.warning("OpenAI API key not found in environment variables")
        return None
    
    # Get index mapping
    client = get_opensearch_client()
    mapping = client.indices.get_mapping(index=index_name)
    properties = mapping[index_name]['mappings'].get('properties', {})
    fields_info = extract_fields_info(properties)
    
    # Step 1: Analyze fields for relevance
    st.write("🔍 **Step 1**: Analyzing field relevance...")
    field_analysis = analyze_fields_for_query(fields_info, question)
    
    if field_analysis:
        st.write("📊 **Field Analysis Complete**:")
        if 'relevant_fields' in field_analysis:
            st.write(f"• Most relevant fields: {', '.join(field_analysis.get('relevant_fields', []))}")
        if 'text_fields' in field_analysis:
            st.write(f"• Text searchable fields: {', '.join(field_analysis.get('text_fields', []))}")
    
    # Rule context
    rule_context = ""
    if rule_description:
        rule_context = f"""
        IMPORTANT CONTEXT: This question relates to the rule description: "{rule_description}"
        Include this rule description in your search strategy.
        """
    
    # Step 2: Generate initial query with deep thinking
    st.write("🧠 **Step 2**: Generating query with deep analysis...")
    
    thinking_agent = AssistantAgent(
        name="DeepQueryThinker",
        model_client=model_client,
        system_message=f"""
        You are an expert OpenSearch query architect that thinks deeply about query construction.
        
        Available fields: {json.dumps(fields_info, indent=2)}
        Field analysis: {json.dumps(field_analysis, indent=2) if field_analysis else "No analysis available"}
        
        Your process:
        1. UNDERSTAND: Break down what the user is really asking for
        2. STRATEGIZE: Determine the best search strategy (exact match, fuzzy search, range, etc.)
        3. PRIORITIZE: Focus on the most relevant fields based on the question
        4. CONSTRUCT: Build an optimized OpenSearch query
        5. VALIDATE: Ensure the query syntax is correct
        
        For text searches, prioritize:
        - Fields likely to contain the relevant text content
        - Use multi_match for searching across multiple text fields
        - Use match_phrase for exact phrase matching when appropriate
        - Use wildcard or fuzzy queries for partial matches
        - Always include the "query" wrapper object.
        
        Return your thinking process AND the final query as JSON:
        {{
            "thinking_process": "Your step-by-step reasoning...",
            "search_strategy": "The strategy you chose and why...",
            "field_selection_reasoning": "Why you selected specific fields...", 
            "query": {{...opensearch query...}}
        }}
        """
    )
    
    deep_prompt = f"""
    Question: {question}
    {rule_context}
    
    Think deeply about this query:
    
    1. What is the user REALLY trying to find?
    2. Which fields are most likely to contain the answer?
    3. What type of search strategy fits best?
    4. How can I make this search both comprehensive and precise?
    
    Focus especially on text-searchable fields for content-based queries.
    
    Provide your complete thinking process and the optimized OpenSearch query.
    """
    
    async def run_deep_thinking():
        result = await thinking_agent.run(task=deep_prompt)
        return result
    
    try:
        result = asyncio.run(run_deep_thinking())
        response = result.messages[1].content
        cleaned_response = clean_json_response(response)
        deep_analysis = json.loads(cleaned_response)
        
        # Display the thinking process
        if 'thinking_process' in deep_analysis:
            st.write("💭 **Thinking Process**:")
            st.write(deep_analysis['thinking_process'])
            
        if 'search_strategy' in deep_analysis:
            st.write("🎯 **Search Strategy**:")
            st.write(deep_analysis['search_strategy'])
            
        if 'field_selection_reasoning' in deep_analysis:
            st.write("📋 **Field Selection**:")
            st.write(deep_analysis['field_selection_reasoning'])
        
        initial_query = deep_analysis.get('query', {})
        
        # Step 3: Iterative refinement
        current_query = initial_query
        
        for iteration in range(max_iterations):
            st.write(f"🔄 **Step {3 + iteration}**: Query refinement iteration {iteration + 1}")
            
            refinement_agent = AssistantAgent(
                name="QueryRefiner",
                model_client=model_client,
                system_message=f"""
                You are a query optimization expert. Review and improve OpenSearch queries.
                
                Focus on:
                1. Ensuring all relevant text fields are included in searches
                2. Optimizing search clauses for better relevance
                3. Adding appropriate filters when beneficial
                4. Balancing precision and recall
                5. Proper handling of nested fields if present
                
                Return JSON with:
                {{
                    "improvements_made": "List of specific improvements...",
                    "confidence_score": 0.0-1.0,
                    "refined_query": {{...improved query...}},
                    "needs_further_refinement": true/false
                }}
                """
            )
            
            refinement_prompt = f"""
            Original question: {question}
            Current query: {json.dumps(current_query, indent=2)}
            Available fields: {json.dumps(fields_info, indent=2)}
            
            Analyze and improve this query. Consider:
            1. Are we searching the right fields?
            2. Is the search strategy optimal?
            3. Are we missing any important text fields?
            4. Can we improve relevance scoring?
            
            Provide specific improvements and the refined query.
            """
            
            async def run_refinement():
                result = await refinement_agent.run(task=refinement_prompt)
                return result
            
            try:
                result = asyncio.run(run_refinement())
                response = result.messages[1].content
                cleaned_response = clean_json_response(response)
                refinement = json.loads(cleaned_response)
                
                if 'improvements_made' in refinement:
                    st.write(f"✨ **Improvements**: {refinement['improvements_made']}")
                
                if 'confidence_score' in refinement:
                    confidence = refinement['confidence_score']
                    st.write(f"📊 **Confidence Score**: {confidence:.2f}")
                
                # Update query if improvements were made
                if 'refined_query' in refinement:
                    current_query = refinement['refined_query']
                
                # Check if we should continue refining
                if not refinement.get('needs_further_refinement', False) or confidence > 0.85:
                    st.write("✅ **Query optimization complete!**")
                    break
                    
            except Exception as e:
                st.write(f"⚠️ Refinement iteration {iteration + 1} failed: {str(e)}")
                break
        
        st.write("🎯 **Final Optimized Query**:")
        st.code(json.dumps(current_query, indent=2))
        
        return current_query
        
    except Exception as e:
        st.error(f"Error in deep research query generation: {str(e)}")
        return None

# Main application logic
def process_question_enhanced(question, selected_index):
    with st.status("Processing your question with deep analysis...", expanded=True) as status:
        st.write(f"🔍 **Using index**: {selected_index}")
        
        # Load rule descriptions and check if question is rule-related
        st.write("📋 **Checking rule relevance**...")
        # descriptions = load_rule_descriptions(question)  # Using your existing function
        descriptions = example_usage(question)
        relevant_rule_description = find_relevant_rule_description(question, descriptions)
        
        if relevant_rule_description:
            st.write(f"✅ **Rule-related question detected!**")
            st.write(f"📄 **Relevant rule**: {relevant_rule_description}")
        else:
            st.write("ℹ️ **General question** (not rule-specific)")
        
        st.write("🚀 **Starting deep research query generation**...")
        
        # Use the enhanced query generation
        query = deep_research_query_generation(question, selected_index, relevant_rule_description)
        
        if query:
            status.update(label="✅ Deep analysis complete!", state="complete")
            return query
        else:
            status.update(label="❌ Failed to generate query", state="error")
            return None


# Function to handle descriptions upload (placeholder for your custom function)
def handle_descriptions_upload():
    """
    Placeholder function for handling descriptions upload.
    Replace this with your custom logic.
    """
    try:
        load_desc = load_rule_descriptions_v2()

        if(load_desc == True ):
            return True
        return False
        
    except Exception as e:
        st.error(f"Error processing descriptions: {str(e)}")
        return False

# Create tabs
st.divider()

# Create two tabs
tab1, tab2 = st.tabs(["💬 Chat Interface", "📤 Upload Descriptions"])

# Tab 1: Chat Interface (existing functionality)
with tab1:
    st.header("Ask Questions About Your Data")
    
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
                        answer = process_question_enhanced(question, selected_index)
                        # st.write("### Answer")
                        # st.write(answer)
            else:
                st.error("No indices found in your OpenSearch cluster. Please create at least one index.")
                
        except Exception as e:
            st.error(f"Error connecting to OpenSearch: {str(e)}")
            st.info("Please ensure all required environment variables are set correctly.")

# Tab 2: Upload Interface
with tab2:
    st.header("Upload Rule Descriptions")
    st.markdown("Process and update rule descriptions in the system.")
    
    # Simple upload button
    if st.button("📤 Upload Descriptions", type="primary", use_container_width=True):
        with st.spinner("Processing descriptions..."):
            success = handle_descriptions_upload()
            if success:
                st.balloons()
                st.success("Rule descriptions have been processed and updated successfully!")
            else:
                st.error("Failed to process the descriptions.")
    

st.divider()