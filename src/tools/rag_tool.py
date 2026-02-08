"""
RAG integration tool for querying the local YogaBharati RAG system.

This module simply calls external RAG services - NO LLMs or embeddings here.
The RAG services handle all the AI/ML processing.

Services:
- RAG API: http://localhost:8080
- RAG UI/Alternative: http://localhost:8501
"""

import os
import sys
import requests
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from sympy import true

from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


# Fix import paths
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from config.settings import settings
    _has_settings = True
except ImportError:
    _has_settings = False
    settings = None

try:
    from .video_indexer import VideoIndexer
except ImportError:
    try:
        from video_indexer import VideoIndexer
    except ImportError:
        VideoIndexer = None


def _get_setting(name: str, default: Any = None) -> Any:
    """Get setting from config.settings, environment, or default."""
    if _has_settings and settings and hasattr(settings, name):
        return getattr(settings, name)
    return os.getenv(name, default)


# RAG Service Configuration
RAG_SERVICE_URL = _get_setting('RAG_SERVICE_URL', 'http://localhost:8080')
RAG_UI_URL = _get_setting('RAG_UI_URL', 'http://localhost:8501')
REQUEST_TIMEOUT = int(_get_setting('RAG_REQUEST_TIMEOUT', 240))

# Lazy initialization for video indexer
_video_indexer: Optional[Any] = None
_videos_info: Optional[list] = None


def get_video_indexer():
    """Lazily initialize and return video indexer and videos info."""
    global _video_indexer, _videos_info
    
    if VideoIndexer is None:
        logger.warning("VideoIndexer not available")
        return None, []
    
    if _video_indexer is None:
        videos_path = os.getenv("VIDEOS_PATH", str(_project_root / "videos"))
        if os.path.isdir(videos_path):
            _video_indexer = VideoIndexer(videos_path)
            _videos_info = _video_indexer.get_videos_info()
            logger.info(f"Initialized video indexer with {len(_videos_info)} videos")
        else:
            logger.warning(f"Videos folder not found: {videos_path}")
            _videos_info = []
    
    return _video_indexer, _videos_info


def check_rag_service() -> bool:
    """Check if the RAG service is accessible."""
    try:
        # Try the main RAG service
        response = requests.get(f"{RAG_SERVICE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    
    try:
        # Try just connecting to the service
        response = requests.get(RAG_SERVICE_URL, timeout=5)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        pass
    
    return False


def query_rag_system(query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the RAG service for yoga class generation.
    
    This function calls the external RAG service at localhost:8080
    which handles all the AI/ML processing (embeddings, retrieval, generation).
    
    Args:
        query_text: User query about yoga classes
        top_k: Number of results to consider (passed to RAG service)
        
    Returns:
        Dictionary with the RAG service response
    """
    logger.info(f"Querying RAG service with: {query_text}")
    
    # Try different endpoint patterns that RAG services commonly use
    endpoints_to_try = [
        (f"{RAG_SERVICE_URL}/query", "POST", {"query": query_text, "top_k": top_k, "temperature": 0.1}),
        # (f"{RAG_SERVICE_URL}/ask", "POST", {"question": query_text, "k": top_k}),
        # (f"{RAG_SERVICE_URL}/chat", "POST", {"message": query_text}),
        # (f"{RAG_SERVICE_URL}/generate", "POST", {"prompt": query_text, "top_k": top_k}),
        # (f"{RAG_SERVICE_URL}/api/query", "POST", {"query": query_text}),
        # (f"{RAG_SERVICE_URL}/api/chat", "POST", {"query": query_text}),
        # (f"{RAG_SERVICE_URL}/v1/query", "POST", {"query": query_text}),
    ]
    
    last_error = None
    
    for endpoint, method, payload in endpoints_to_try:
        try:
            logger.debug(f"Trying endpoint: {endpoint}")
            
            if method == "POST":
                response = requests.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=REQUEST_TIMEOUT
                )
            else:
                response = requests.get(
                    endpoint,
                    params=payload,
                    timeout=REQUEST_TIMEOUT
                )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"RAG service responded successfully from {endpoint}")
                
                # Normalize response format
                return _normalize_rag_response(data, query_text)
            
            elif response.status_code == 404:
                # Endpoint not found, try next
                continue
            else:
                logger.warning(f"RAG service returned {response.status_code} from {endpoint}")
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.ConnectionError:
            last_error = f"Cannot connect to RAG service at {endpoint}"
            continue
        except requests.exceptions.Timeout:
            last_error = f"Timeout connecting to {endpoint}"
            continue
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            continue
        except ValueError as e:
            # JSON decode error
            last_error = f"Invalid JSON response: {e}"
            continue
    
    # All endpoints failed
    logger.error(f"All RAG service endpoints failed. Last error: {last_error}")
    return {
        "response": "",
        "context": "",
        "error": last_error or "Could not connect to RAG service",
        "query": query_text,
        "results": []
    }


def _normalize_rag_response(data: Any, query_text: str) -> Dict[str, Any]:
    """Normalize different RAG service response formats to a standard format."""
    
    # Handle string response
    if isinstance(data, str):
        return {
            "response": data,
            "context": data,
            "query": query_text,
            "results": []
        }
    
    # Handle dict response - try common field names
    if isinstance(data, dict):
        response_text = (
            data.get("response") or 
            data.get("answer") or 
            data.get("result") or 
            data.get("output") or
            data.get("generated_text") or
            data.get("text") or
            data.get("content") or
            data.get("message") or
            ""
        )
        
        context = (
            data.get("context") or
            data.get("sources") or
            data.get("documents") or
            data.get("retrieved") or
            ""
        )
        
        # If context is a list, join it
        if isinstance(context, list):
            context = "\n\n".join(str(c) for c in context)
        
        results = data.get("results") or data.get("hits") or []
        
        return {
            "response": response_text,
            "context": context if context else response_text,
            "query": query_text,
            "results": results,
            "raw": data  # Keep original for debugging
        }
    
    # Handle list response
    if isinstance(data, list):
        combined = "\n\n".join(str(item) for item in data)
        return {
            "response": combined,
            "context": combined,
            "query": query_text,
            "results": data
        }
    
    # Fallback
    return {
        "response": str(data),
        "context": str(data),
        "query": query_text,
        "results": []
    }


def search_yoga_class_details(class_type: str) -> List[Dict]:
    """
    Search for detailed yoga class information via RAG service.
    
    Args:
        class_type: Type of yoga class (e.g., "beginner", "advanced", "meditation")
        
    Returns:
        List of relevant yoga class documents
    """
    logger.info(f"Searching for {class_type} yoga class details")
    
    query = f"Generate a detailed {class_type} yoga class with specific asanas, durations, and instructions"
    result = query_rag_system(query, top_k=3)
    
    return result.get("results", [])


def search_yogabharati_videos(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search for YogaBharati videos from the local video library.
    
    This searches the local videos folder, NOT the RAG service.
    
    Args:
        query: Search query for yoga videos (e.g., "Bhujangasana")
        max_results: Maximum number of results to return
        
    Returns:
        List of matching video metadata
    """
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
        # Check filename and any available metadata
        searchable_text = " ".join([
            str(video.get('title', '')),
            str(video.get('filename', '')),
            str(video.get('name', '')),
            str(video.get('subtitle', '')),
            str(video.get('comment', '')),
            str(video.get('genre', '')),
            str(video.get('tags', '')),
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


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("RAG Tool Test - External RAG Service")
    print("=" * 60)
    
    print(f"\nRAG Service URL: {RAG_SERVICE_URL}")
    print(f"RAG UI URL: {RAG_UI_URL}")
    
    # Test RAG service connection
    print("\n--- Testing RAG Service Connection ---")
    if check_rag_service():
        print(f"✅ RAG service is accessible at {RAG_SERVICE_URL}")
    else:
        print(f"❌ Cannot connect to RAG service at {RAG_SERVICE_URL}")
        print("   Make sure your RAG service is running!")
    
    # Test RAG query
    print("\n--- Testing RAG Query ---")
    try:
        result = query_rag_system("Create a beginner yoga class with Bhujangasana", top_k=3)
        if result.get("error"):
            print(f"❌ RAG query error: {result['error']}")
        else:
            print(f"✅ RAG query successful")
            response_preview = result.get('response', '')[:300]
            print(f"   Response preview: {response_preview}...")
    except Exception as e:
        print(f"❌ RAG query error: {e}")
    
    # Test video search
    print("\n--- Testing Video Search ---")
    try:
        videos = search_yogabharati_videos("Bhujangasana")
        print(f"✅ Found {len(videos)} videos")
        for v in videos[:3]:
            print(f"   - {v.get('filename', v.get('title', 'Unknown'))}")
    except Exception as e:
        print(f"❌ Video search error: {e}")