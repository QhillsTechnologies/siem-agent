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
    Returns both must-have fields and most relevant fields based on question analysis.
    """
    agent = AssistantAgent(
        name="FieldAnalyzer",
        model_client=model_client,
        system_message=f"""
        You are a database field selection expert. Your task is to analyze user questions and categorize fields into two distinct categories based on query requirements.
        
        Available fields and their types:
        {json.dumps(fields_info, indent=2)}
        
        FIELD CATEGORIZATION FRAMEWORK:
        
        MUST_FIELDS - Fields that represent EXACT REQUIREMENTS:
        - Fields containing specific identifiers, codes, names, or exact values explicitly mentioned in the question
        - Fields that represent mandatory filtering criteria with concrete values
        - Fields that must match exactly for the query to be meaningful
        - Leave EMPTY if the question contains no specific identifiers or exact values
        
        MOST_RELEVANT_FIELDS - Fields that contain the PRIMARY INFORMATION needed:
        - Fields that hold the core data required to answer the question
        - Fields that provide essential context for a complete response  
        - Fields that directly relate to the information being sought
        - Always select exactly THREE fields for optimal relevance scoring
        
        SELECTION PRINCIPLES:
        - MUST_FIELDS: Only populate when question contains concrete, specific references
        - MOST_RELEVANT_FIELDS: Focus on fields containing the actual answer data
        - All selected fields must exist in the provided schema
        - Prioritize fields that directly address the user's information needs
        - Be extremely strict about MUST_FIELDS - only include for explicit value matches
        
        OUTPUT FORMAT:
        Return a JSON object with exactly this structure:
        {{
            "MUST_FIELDS": ["field1", "field2"],
            "MOST_RELEVANT_FIELDS": ["field1", "field2", "field3"]
        }}
        """
    )
    
    prompt = f"""
    User Question: "{question}"
    
    ANALYSIS PROCESS:
    
    1. QUESTION DECOMPOSITION:
    - Identify what specific information the user is requesting
    - Detect any explicit identifiers, codes, names, or exact values mentioned
    - Determine what type of data would constitute a complete answer
    
    2. MUST_FIELDS DETERMINATION:
    - Scan for SPECIFIC references to identifiers, codes, names, or exact values
    - Map explicit terms in the question to corresponding field names
    - Only include fields when question contains concrete, specific values
    - If question is general or conceptual, keep MUST_FIELDS empty
    
    3. MOST_RELEVANT_FIELDS SELECTION:
    - Identify fields containing the primary information needed for the answer
    - Select fields that provide necessary context and details
    - Choose exactly three fields that best address the question requirements
    - Focus on fields that contain the actual answer data
    
    4. VALIDATION:
    - Ensure all selected fields exist in the provided schema
    - Verify field selections align with question requirements
    - Maintain strict criteria for MUST_FIELDS inclusion
    
    Analyze the question and return the JSON response with both field categories.
    """
    
    async def run_analysis():
        result = await agent.run(task=prompt)
        return result
    
    try:
        result = asyncio.run(run_analysis())
        response = result.messages[1].content
        cleaned_response = clean_json_response(response)
        analysis = json.loads(cleaned_response)
        
        # Extract both must fields and most relevant fields from the original fields_info
        must_fields = analysis.get('MUST_FIELDS', [])
        most_relevant_fields = analysis.get('MOST_RELEVANT_FIELDS', [])
        
        # Combine and create relevant fields info
        all_selected_fields = list(set(must_fields + most_relevant_fields))
        relevant_fields_info = {
            field: fields_info[field] 
            for field in all_selected_fields 
            if field in fields_info
        }
        
        # Return both the categorized fields and the field info
        return {
            'categorized_fields': analysis,
            'relevant_fields_info': relevant_fields_info
        }
        
    except Exception as e:
        st.error(f"Error analyzing fields: {str(e)}")
        return None


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
    print("field_analysis:  ",field_analysis)
    
    if field_analysis:
        st.write("📊 **Field Analysis Complete**:")
        categorized_fields = field_analysis.get('categorized_fields', {})
        must_fields = categorized_fields.get('MUST_FIELDS', [])
        relevant_fields = categorized_fields.get('MOST_RELEVANT_FIELDS', [])
        
        if must_fields:
            st.write(f"• Must-have fields: {', '.join(must_fields)}")
        if relevant_fields:
            st.write(f"• Most relevant fields: {', '.join(relevant_fields)}")
    
    # Rule context
    rule_context = ""
    if rule_description:
        rule_context = f"""
        RULE SPECIFICATION: A specific rule is provided: "{rule_description}"
        When rule_description is provided, create exact match query for rule.description field.
        Use term query for precise rule matching without wildcards or complex clauses.
        """
        print('rule_content: ',rule_context)
    
    # Step 2: Generate initial query with deep thinking
    st.write("🧠 **Step 2**: Generating query with deep analysis...")
    
    thinking_agent = AssistantAgent(
        name="DeepQueryThinker",
        model_client=model_client,
        system_message=f"""
        You are an OpenSearch query builder that creates bool queries based on field categorization analysis.

        **Field analysis:** {json.dumps(field_analysis, indent=2) if field_analysis else "No analysis available"}

        **QUERY CONSTRUCTION LOGIC:**
        
        MUST CLAUSE USAGE:
        - Only Include MUST_FIELDS in the "must" clause for exact matching requirements
        - Use when specific identifiers or exact values need to be matched
        - These are mandatory conditions that must be satisfied
        
        SHOULD CLAUSE USAGE:
        - Include MOST_RELEVANT_FIELDS in the "should" clause for relevance scoring
        - Always set "minimum_should_match": 1 when using should clauses
        - These conditions improve relevance but are not mandatory
        
        QUERY TYPE SELECTION:
        - term: For exact identifiers, IDs, and categorical values
        - match: For flexible text searches with analysis
        - range: For numerical thresholds and date ranges
        
        FIELD NAME HANDLING:
        - Use exact field names from field analysis without modification
        - Do not add or remove any suffixes from field names
        - Use field names exactly as they appear in the mapping

        STRUCTURE REQUIREMENTS:
        - MUST start with {{"query": {{"bool": {{...}}}}}} structure
        - Use bool query structure with appropriate must/should clauses
        - Only use fields that exist in the field analysis
        - Do not include sort or size parameters
        - When using should clauses, always include "minimum_should_match": 1
        
        **Return JSON with:**
        {{
            "thinking_process": "Brief explanation of your query structure approach",
            "query": {{your OpenSearch query}}
        }}
        """
    )
    
    deep_prompt = f"""
    **Question:** {question}  
    **Field Analysis:** {json.dumps(field_analysis, indent=2) if field_analysis else "No analysis"}
    **Rule description:** {rule_description if rule_description else "None"}

    Create an OpenSearch bool query following this structure logic:
    
    1. **MUST CLAUSE CONSTRUCTION:**
    - If MUST_FIELDS are present, then and then only place them in the "must" clause
    - Use appropriate query types for exact matching
    - These represent mandatory filtering conditions
    
    2. **SHOULD CLAUSE CONSTRUCTION:**
    - Place MOST_RELEVANT_FIELDS in the "should" clause
    - Always include "minimum_should_match": 1
    - These improve relevance scoring for better results
    
    3. **QUERY TYPE DETERMINATION:**
    - Analyze field content to select optimal query type
    - Use term for exact values, match for text, range for numbers
    - Ensure field names match exactly those in field analysis
    
    4. **RULE INTEGRATION:**
    - When rule_description is provided, Always include exact match condition in should clause
    - Use term query for "rule.description" field matching
    
    5. **VALIDATION:**
    - Only use field names present in the field analysis
    - MUST start with {{"query": {{"bool": {{...}}}}}} structure
    - Include "minimum_should_match" : 1 when using should clauses
    
    Build the query structure based on the field categorization and question intent.
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
        
        initial_query = deep_analysis.get('query', {})
        
        # Step 3: Iterative refinement
        current_query = initial_query
        
        for iteration in range(max_iterations):
            st.write(f"🔄 **Step {3 + iteration}**: Query refinement iteration {iteration + 1}")
            
            refinement_agent = AssistantAgent(
                name="QueryRefiner",
                model_client=model_client,
                system_message=f"""
                You optimize OpenSearch bool queries by ensuring proper must/should clause structure and field usage.

                **Field analysis:** {json.dumps(field_analysis, indent=2)}

                **OPTIMIZATION REQUIREMENTS:**
                
                MUST CLAUSE OPTIMIZATION:
                - Ensure Only MUST_FIELDS are properly placed in "must" clause
                - Use appropriate query types for exact matching
                - Validate these represent mandatory conditions
                
                SHOULD CLAUSE OPTIMIZATION:
                - Ensure MOST_RELEVANT_FIELDS are in "should" clause
                - Always include "minimum_should_match": 1
                - Optimize for relevance scoring
                
                QUERY TYPE OPTIMIZATION:
                - term: For exact identifiers and categorical values
                - match_phrase: For descriptive text matching
                - match: For flexible text searches
                - range: For numerical thresholds and dates
                
                STRUCTURE VALIDATION:
                - Maintain proper bool query structure
                - MUST start with {{"query": {{"bool": {{...}}}}}} structure
                - Validate all fields exist in field analysis
                - Remove redundant or conflicting clauses
                - Ensure minimum_should_match is set when using should clauses
                
                **Return JSON:**
                {{
                    "improvements_made": "Description of optimizations applied",
                    "confidence_score": 0.0-1.0,
                    "refined_query": {{optimized query}},
                    "needs_further_refinement": true/false
                }}
                """
            )
            
            refinement_prompt = f"""
            Question: {question}
            Current query: {json.dumps(current_query, indent=2)}
            Field analysis: {json.dumps(field_analysis, indent=2) if field_analysis else "No analysis"}
            Rule description: {rule_description if rule_description else "None"}
            
            Optimize this OpenSearch query by:
            
            1. **MUST CLAUSE VALIDATION:**
            - Ensure MUST_FIELDS are correctly placed in "must" clause
            - Verify appropriate query types for exact matching
            - Confirm mandatory conditions are properly structured
            
            2. **SHOULD CLAUSE VALIDATION:**
            - Ensure MOST_RELEVANT_FIELDS are in "should" clause
            - Verify "minimum_should_match": 1 is included 
            
            3. **QUERY TYPE OPTIMIZATION:**
            - Validate query types match field content and purpose
            - Ensure optimal matching behavior for each field
            - Remove inefficient query constructions

            4. **RULE INTEGRATION:**
            - When rule description is provided, Always include exact match condition in should clause
            - Use term query for rule.description field matching
            
            5. **STRUCTURE REFINEMENT:**
            - MUST start with {{"query": {{"bool": {{...}}}}}} structure
            - Eliminate redundant or conflicting clauses
            - Ensure all fields exist in field analysis
            - Keep focus on user's specific question intent
            
            Apply optimizations to improve query effectiveness and structure.
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
        
        # Final validation
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