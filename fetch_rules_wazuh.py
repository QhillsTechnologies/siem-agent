import requests
import json
import os
from typing import List, Optional, Dict, Any
import time

def authenticate_wazuh_api() -> Optional[str]:
    """
    Authenticates with the Wazuh API and returns the access token.
    
    Returns:
        str: Access token if authentication successful, None otherwise
    """
    
    # Get configuration from environment variables
    base_url = os.getenv('WAZUH_BASE_URL')
    username = os.getenv('WAZUH_USERNAME')
    password = os.getenv('WAZUH_PASSWORD')
    
    # Check if all required environment variables are set
    if not all([base_url, username, password]):
        missing = []
        if not base_url: missing.append('WAZUH_BASE_URL')
        if not username: missing.append('WAZUH_USERNAME')
        if not password: missing.append('WAZUH_PASSWORD')
        print(f"Missing required environment variables: {', '.join(missing)}")
        return None
    
    # Authentication API configuration
    auth_url = f"{base_url.rstrip('/')}/security/user/authenticate"

    # Authentication credentials as URL parameters
    auth_params = {
        "user": username,
        "password": password
    }
    
    # Headers for authentication request
    auth_headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the authentication request with params
        response = requests.post(
            auth_url, 
            auth=(username,password), 
            headers=auth_headers 
        )
        
        # Check if authentication was successful
        if response.status_code == 200:
            data = response.json()
            token = data.get("data", {}).get("token")
            
            if token:
                print("Authentication successful")
                return token
            else:
                print("Authentication failed: No token in response")
                return None
                
        else:
            print(f"Authentication failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error during authentication: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing authentication response: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error during authentication: {str(e)}")
        return None

def fetch_wazuh_rules_descriptions() -> Optional[List[str]]:
    """
    Fetches all rule descriptions from the Wazuh API after authentication.
    
    Returns:
        List[str]: List of rule descriptions, or None if there was an error
    """
    
    # First, authenticate and get the token
    token = authenticate_wazuh_api()
    if not token:
        print("Failed to authenticate. Cannot fetch rules.")
        return None
    
    # Get base URL from environment variable
    base_url = os.getenv('WAZUH_BASE_URL')
    if not base_url:
        print("Missing WAZUH_BASE_URL environment variable")
        return None
    
    # API configuration for rules
    api_url = f"{base_url.rstrip('/')}/rules"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.get(api_url, headers=headers, verify=False)
        
        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract descriptions from the response
            descriptions = []
            affected_items = data.get("data", {}).get("affected_items", [])
            print(f"Successfully fetched {len(affected_items)} affected_items")
            
            for item in affected_items:
                description = item.get("description", "")
                if description:  # Only add non-empty descriptions
                    descriptions.append(description)
            
            print(f"Successfully fetched {len(descriptions)} rule descriptions")
            return descriptions
            
        else:
            print(f"Rules API request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error making rules API request: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None


# def fetch_wazuh_rules_descriptions() -> Optional[List[str]]:
#     """
#     Enhanced version with retry logic for failed batches.
    
#     Returns:
#         List[str]: List of rule descriptions, or None if there was an error
#     """
    
#     # First, authenticate and get the token
#     token = authenticate_wazuh_api()
#     if not token:
#         print("Failed to authenticate. Cannot fetch rules.")
#         return None
    
#     # Get base URL from environment variable
#     base_url = os.getenv('WAZUH_BASE_URL')
#     if not base_url:
#         print("Missing WAZUH_BASE_URL environment variable")
#         return None
    
#     # API configuration for rules
#     api_url = f"{base_url.rstrip('/')}/rules"
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }
    
#     # Pagination settings
#     total_entries = 6000
#     batch_size = 500
#     current_offset = 0
#     all_descriptions = []
#     max_retries = 3
    
#     try:
#         # Calculate number of iterations needed
#         num_iterations = (total_entries + batch_size - 1) // batch_size
#         print(f"Fetching up to {total_entries} rules in batches of {batch_size}")
#         print(f"Total iterations planned: {num_iterations}")
        
#         for iteration in range(num_iterations):
#             # Set parameters for current iteration
#             params = {
#                 'offset': current_offset,
#                 'limit': batch_size,
#                 'pretty': False,
#                 'wait_for_complete': False
#             }
            
#             print(f"Iteration {iteration + 1}/{num_iterations} - Offset: {current_offset}")
            
#             # Retry logic for current batch
#             success = False
#             for retry in range(max_retries):
#                 try:
#                     # Make the API request
#                     response = requests.get(
#                         api_url, 
#                         headers=headers, 
#                         params=params, 
#                         verify=False, 
#                         timeout=30
#                     )
                    
#                     # Check if request was successful
#                     if response.status_code == 200:
#                         data = response.json()
                        
#                         # Extract affected_items from the response
#                         affected_items = data.get("data", {}).get("affected_items", [])
#                         print(f"Fetched {len(affected_items)} affected_items in this batch")
                        
#                         # Extract descriptions from current batch
#                         batch_descriptions = []
#                         for item in affected_items:
#                             description = item.get("description", "")
#                             if description:  # Only add non-empty descriptions
#                                 batch_descriptions.append(description)
                        
#                         # Add current batch descriptions to total
#                         all_descriptions.extend(batch_descriptions)
#                         print(f"Added {len(batch_descriptions)} descriptions (Total so far: {len(all_descriptions)})")
                        
#                         success = True
                        
#                         # Break if we got fewer results than expected (reached end of data)
#                         if len(affected_items) < batch_size:
#                             print("Reached end of data (received fewer results than batch size)")
#                             return all_descriptions if all_descriptions else None
                        
#                         break  # Success, exit retry loop
                        
#                     else:
#                         print(f"API request failed with status code: {response.status_code}")
#                         if retry < max_retries - 1:
#                             print(f"Retrying... (attempt {retry + 2}/{max_retries})")
#                             time.sleep(2 ** retry)  # Exponential backoff
#                         else:
#                             print(f"Max retries exceeded for offset {current_offset}")
                            
#                 except requests.exceptions.RequestException as e:
#                     print(f"Request error on attempt {retry + 1}: {str(e)}")
#                     if retry < max_retries - 1:
#                         print(f"Retrying... (attempt {retry + 2}/{max_retries})")
#                         time.sleep(2 ** retry)  # Exponential backoff
#                     else:
#                         print(f"Max retries exceeded for offset {current_offset}")
            
#             if not success:
#                 print(f"Failed to fetch batch at offset {current_offset} after {max_retries} attempts")
#                 # You can choose to break or continue to next batch
#                 # For now, we'll continue to try the next batch
                
#             # Update offset for next iteration
#             current_offset += batch_size
            
#             # Small delay between successful requests
#             if success:
#                 time.sleep(0.1)
        
#         print(f"Successfully fetched {len(all_descriptions)} total rule descriptions")
#         print("descriptions",all_descriptions)
#         return all_descriptions if all_descriptions else None
        
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         return None



