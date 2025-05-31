import requests
import json
import os
from typing import List, Optional, Dict, Any

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

