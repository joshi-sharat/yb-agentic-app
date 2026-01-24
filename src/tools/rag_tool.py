"""
RAG integration tool for querying the local YogaBharati RAG system.
"""
import logging
import os
from typing import Optional

from openai import AzureOpenAI
from opensearchpy import OpenSearch, exceptions as os_exceptions

from config.settings import settings
from .video_indexer import VideoIndexer

logger = logging.getLogger(__name__)

# Lazy initialization for video indexer
_video_indexer: Optional[VideoIndexer] = None
_videos_info: Optional[list] = None


def get_video_indexer():
    """Lazily initialize and return video indexer and videos info."""
    global _video_indexer, _videos_info
    if _video_indexer is None:
        videos_path = os.getenv("VIDEOS_PATH", "videos")
        if os.path.isdir(videos_path):
            _video_indexer = VideoIndexer(videos_path)
            _videos_info = _video_indexer.get_videos_info()
            logger.info(f"Initialized video indexer with {len(_videos_info)} videos")
        else:
            logger.warning(f"Videos folder not found: {videos_path}")
            _videos_info = []
    return _video_indexer, _videos_info


def get_openai_client() -> AzureOpenAI:
    """Get Azure OpenAI client instance."""
    return AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version="2024-10-21",
        azure_endpoint=settings.API_ENDPOINT
    )


def get_opensearch_client() -> OpenSearch:
    """Get OpenSearch client instance."""
    return OpenSearch(
        hosts=[{
            'host': settings.OPENSEARCH_HOST,
            'port': settings.OPENSEARCH_PORT
        }],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False
    )


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for text using Azure OpenAI.
    
    Args:
        texts: List of text strings to generate embeddings for
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        client = get_openai_client()
        embeddings = []
        
        # Process in batches to avoid token limits
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=settings.EMBEDDING_MODEL_ID  # e.g., "text-embedding-ada-002"
            )
            for item in response.data:
                embeddings.append(item.embedding)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        # Return zero vectors as fallback
        return [[0.0] * 1536 for _ in texts]


def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    top_k: int = 5
) -> list[dict]:
    """
    Perform hybrid search in OpenSearch combining text and vector search.
    
    Args:
        query_text: The text query for keyword search
        query_embedding: The embedding vector for semantic search
        top_k: Number of results to return
        
    Returns:
        List of search hits with scores
    """
    try:
        os_client = get_opensearch_client()
        
        # Hybrid query combining BM25 text search and kNN vector search
        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": {
                                    "query": query_text,
                                    "boost": 1.0
                                }
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        response = os_client.search(
            index=settings.OPENSEARCH_INDEX,
            body=query
        )
        
        hits = response.get('hits', {}).get('hits', [])
        logger.info(f"Hybrid search returned {len(hits)} results")
        return hits
        
    except os_exceptions.NotFoundError:
        logger.warning(f"OpenSearch index '{settings.OPENSEARCH_INDEX}' not found")
        return []
    except os_exceptions.ConnectionError:
        logger.error("Could not connect to OpenSearch")
        return []
    except Exception as e:
        logger.error(f"Error performing hybrid search: {str(e)}")
        return []


def query_rag_system(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Query the local RAG system for relevant yoga class information.

    Args:
        query_text: User query about yoga classes
        top_k: Number of top results to return (default: 5)

    Returns:
        List of relevant documents from the RAG system
    """
    try:
        logger.info(f"Querying RAG system with: {query_text}")

        # Generate embeddings for the query
        query_embedding = generate_embeddings([query_text])[0]

        # Perform hybrid search in OpenSearch
        results = hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Format results
        formatted_results = []
        for hit in results:
            source = hit.get("_source", {})
            formatted_result = {
                "doc_id": hit.get("_id", ""),
                "text": source.get("text", ""),
                "document_name": source.get("document_name", ""),
                "score": hit.get("_score", 0),
            }
            formatted_results.append(formatted_result)

        logger.info(f"RAG query returned {len(formatted_results)} results")
        return formatted_results

    except Exception as e:
        logger.error(f"Error querying RAG system: {str(e)}")
        return []


def search_yoga_class_details(class_type: str) -> list[dict]:
    """
    Search for detailed yoga class information in the RAG system.

    Args:
        class_type: Type of yoga class (e.g., "beginner", "advanced", "meditation")

    Returns:
        List of relevant yoga class documents
    """
    try:
        logger.info(f"Searching for {class_type} yoga class details")

        query = f"yoga class {class_type} instructions benefits"
        results = query_rag_system(query, top_k=3)

        return results

    except Exception as e:
        logger.error(f"Error searching yoga class details: {str(e)}")
        return []


def search_yogabharati_videos(query: str, max_results: int = 10) -> list[dict]:
    """
    Search for YogaBharati videos from the local video library.
    
    Args:
        query: Search query for yoga videos
        max_results: Maximum number of results to return
        
    Returns:
        List of matching video metadata
    """
    try:
        logger.info(f"Searching local videos for: {query}")
        
        _, videos_info = get_video_indexer()
        
        if not videos_info:
            logger.warning("No videos available in the library")
            return []
        
        # Simple keyword matching on video metadata
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        matched_videos = []
        for video in videos_info:
            # Check title, subtitle, comment, and tags
            searchable_text = " ".join([
                str(video.get('title', '')),
                str(video.get('subtitle', '')),
                str(video.get('comment', '')),
                str(video.get('genre', '')),
                str(video.get('other_tags', ''))
            ]).lower()
            
            # Calculate simple relevance score
            score = sum(1 for term in query_terms if term in searchable_text)
            
            if score > 0:
                matched_videos.append({
                    **video,
                    'relevance_score': score
                })
        
        # Sort by relevance and limit results
        matched_videos.sort(key=lambda x: x['relevance_score'], reverse=True)
        results = matched_videos[:max_results]
        
        logger.info(f"Found {len(results)} matching videos")
        return results
        
    except Exception as e:
        logger.error(f"Error searching videos: {str(e)}")
        return []
