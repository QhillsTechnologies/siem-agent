import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Tuple
from openai import OpenAI
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
import uuid

class OpenAIEmbeddingFunction(Embeddings):
    """Custom embedding function for Chroma using OpenAI's text-embedding-ada-002."""
    def __init__(self, client: OpenAI, model_name: str = "text-embedding-ada-002"):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error generating embeddings for documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding for query: {str(e)}")
            raise

class OpenAISemanticMatcher:
    def __init__(self, api_key=None,  persist_directory: str = "./chroma_db", collection_name: str = "description_data"):
        """
        Initialize the OpenAI semantic similarity matcher with text-embedding-ada-002.

        Args:
            api_key (str): OpenAI API key (can also use environment variable)
        """
        try:
            api_key = api_key or os.getenv('OPENAI_API_KEY_V2')
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY_V2 environment variable or pass api_key parameter")

            self.client = OpenAI(api_key=api_key)
            self.model_name = 'text-embedding-ada-002'
            self.persist_directory = persist_directory
            self.collection_name = 'description_data'
            self.embedding_function = OpenAIEmbeddingFunction(client=self.client, model_name=self.model_name)
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=persist_directory,

                
            )

        except ImportError:
            raise ImportError("Install openai: pip install openai")

    @st.cache_resource
    def _get_cached_embeddings(_self, string_array: List[str]) -> np.ndarray:
        """
        Compute and cache embeddings for the input strings in batches.
        """
        print("cache_embedding", len(string_array))
        
        batch_size = 1000  # Adjust based on your needs
        all_embeddings = []
        
        for i in range(0, len(string_array), batch_size):
            batch = string_array[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(string_array) + batch_size - 1)//batch_size}")
            
            try:
                response = _self.client.embeddings.create(
                    input=batch,
                    model=_self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # You might want to implement retry logic here
                raise
        
        return np.array(all_embeddings)
    
    def create_store_descriptions(self, string_array: List[str]) -> tuple[np.ndarray, List[str]]:
        """
        Compute embeddings for the input strings in batches and store them in Chroma DB.
        
        Args:
            string_array: List of strings to generate and store embeddings for
        Returns:
            Tuple of (numpy array of embeddings, list of document IDs)
        """
        print("cache_embedding", len(string_array))
        
        batch_size = 1000  # Adjust based on your needs
        all_embeddings = []
        all_doc_ids = []
        documents = string_array  # Store original strings
        
        # Generate unique IDs for each document
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Convert strings to LangChain Document objects
        from langchain_core.documents import Document
        doc_objects = [Document(page_content=doc) for doc in documents]
        
        for i in range(0, len(string_array), batch_size):
            batch = string_array[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(string_array) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [item.embedding for item in response.data]
                
                # Add batch to Chroma DB
                self.vector_store._collection.add(
                    documents=batch,
                    embeddings=batch_embeddings,
                    ids=batch_ids
                )
                
                all_embeddings.extend(batch_embeddings)
                all_doc_ids.extend(batch_ids)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                raise
        
        # Persist the vector store
        self.vector_store.persist()
        print(f"Added {len(documents)} documents to Chroma DB.")
        
        return np.array(all_embeddings)

    def find_top_similar(self, query_string: str, top_k: int = 200) -> List[str]:
        """
        Find top K most semantically similar strings to the query string using embeddings from Chroma DB.

        Args:
            query_string: The input string to match against
            top_k: Number of top matches to return (default: 20)

        Returns:
            List of top K similar strings
        """
        try:
            # Get embedding for the query string
            query_response = self.client.embeddings.create(
                input=[query_string],
                model=self.model_name
            )
            query_embedding = [query_response.data[0].embedding]

            # Perform similarity search in Chroma DB
            results = self.vector_store.similarity_search_with_score(
                query=query_string,
                k=top_k
            )

            # Extract document contents
            valid_strings = [doc.page_content for doc, _ in results]

            if not valid_strings:
                print("No similar documents found in Chroma DB.")
                return []

            return valid_strings

        except Exception as e:
            print(f"Error in find_top_similar: {str(e)}")
            return []

# Example usage
def example_usage(query_string):
    print("=== OpenAI Semantic Similarity Matching ===\n")

    try:
        # Initialize matcher
        matcher = OpenAISemanticMatcher()

        # Find top 20 similar strings
        results = matcher.find_top_similar(query_string, top_k=200)

        print("RESULTS: ", results)
      
        return results
    
    except Exception as e:
        print(f"Error: {e}")
        return []