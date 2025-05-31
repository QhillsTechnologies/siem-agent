import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Tuple
from openai import OpenAI

class OpenAISemanticMatcher:
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI semantic similarity matcher with text-embedding-ada-002.

        Args:
            api_key (str): OpenAI API key (can also use environment variable)
        """
        try:
            api_key = api_key or os.getenv('OPENAI_API_KEY_V2')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")

            self.client = OpenAI(api_key=api_key)
            self.model_name = 'text-embedding-ada-002'

        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def find_top_similar(self, string_array: List[str], query_string: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Find top K most semantically similar strings to the query string using OpenAI embeddings.

        Args:
            string_array: List of strings to search through
            query_string: The input string to match against
            top_k: Number of top matches to return (default: 20)

        Returns:
            List of tuples (string, similarity_score) sorted by similarity
        """
        # Get embeddings for all strings
        all_strings = string_array + [query_string]

        # Batch API call for efficiency
        response = self.client.embeddings.create(
            input=all_strings,
            model=self.model_name
        )

        # Extract embeddings using correct syntax: response.data[i].embedding
        embeddings = np.array([item.embedding for item in response.data])

        query_embedding = embeddings[-1].reshape(1, -1)
        document_embeddings = embeddings[:-1]

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results with similarity scores
        return [(string_array[i], similarities[i]) for i in top_indices]

# Example usage
def example_usage(string_array, query_string):
    # Sample log data
    

    print("=== OpenAI Semantic Similarity Matching ===\n")

    try:
        # Initialize matcher
        matcher = OpenAISemanticMatcher()

        # Find top 20 similar strings
        results = matcher.find_top_similar(string_array, query_string, top_k=20)
        final_results =[]

        for i, (string, score) in enumerate(results, 1):
            print(f"{i:2d}. {string:<45} | Score: {score:.4f}")
            final_results.append(string)
        return final_results
    
    except Exception as e:
        print(f"Error: {e}")
    

