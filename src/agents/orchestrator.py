"""
YogaBharati Agentic Orchestrator - Main Agent Implementation

This orchestrator:
1. Receives user query
2. Calls the RAG service to get a generated yoga class
3. Parses the response for Asana names
4. Matches Asana names with local video files
5. Formats the response with video embeds
6. Returns the enhanced response

NO LLM here - the RAG service handles all AI/ML processing.
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Optional, List, Dict
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


# Fix import paths
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # src/agents/orchestrator.py -> project root
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Import RAG tool
try:
    from src.tools import (
        search_yogabharati_videos,
        query_rag_system,
        search_yoga_class_details,
    )
except ImportError:
    try:
        from src.tools import (
            search_yogabharati_videos,
            query_rag_system,
            search_yoga_class_details,
        )
    except ImportError:
        from src.tools import (
            search_yogabharati_videos,
            query_rag_system,
            search_yoga_class_details,
        )

# Path to videos folder
VIDEOS_FOLDER = _project_root / "videos"


class VideoMatcher:
    """Utility class to match Asana names with local video files."""

    # Common yoga asana names to look for in text
    KNOWN_ASANAS = [
        "bhujangasana", "cobra",
        "trikonasana", "triangle",
        "tadasana", "mountain",
        "vrikshasana", "tree",
        "virabhadrasana", "warrior",
        "uttanasana", "forward fold",
        "adho mukha svanasana", "downward dog",
        "balasana", "child",
        "savasana", "shavasana", "corpse",
        "padmasana", "lotus",
        "sukhasana", "easy pose",
        "vajrasana", "thunderbolt",
        "matsyasana", "fish",
        "halasana", "plow",
        "sarvangasana", "shoulder stand",
        "dhanurasana", "bow",
        "ustrasana", "camel",
        "paschimottanasana", "seated forward",
        "setu bandhasana", "bridge",
        "marjariasana", "cat",
        "bitilasana", "cow",
        "navasana", "boat",
        "surya namaskar", "suryanamaskar", "sun salutation",
        "pranayama", "breathing",
        "kapalbhati", "kapalabhati",
        "anulom vilom", "alternate nostril",
        "naukasana", "boat pose",
        "chakrasana", "wheel",
        "garudasana", "eagle",
        "gomukhasana", "cow face",
        "ardha chandrasana", "half moon",
        "parivrtta", "twisted",
        "utkatasana", "chair",
        "malasana", "garland", "squat",
    ]

    def __init__(self, videos_path: Path = VIDEOS_FOLDER):
        """Initialize the video matcher with the videos folder path."""
        self.videos_path = Path(videos_path)
        self.video_files: Dict[str, Path] = {}
        self._index_videos()

    def _index_videos(self) -> None:
        """Index all video files in the videos folder."""
        if not self.videos_path.exists():
            logger.warning(f"Videos folder not found: {self.videos_path}")
            return

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.MP4', '.MOV'}

        for video_file in self.videos_path.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in {ext.lower() for ext in video_extensions}:
                # Extract asana name from filename
                asana_name = self._extract_asana_name(video_file.stem)
                self.video_files[asana_name.lower()] = video_file
                logger.debug(f"Indexed video: {asana_name} -> {video_file}")

        logger.info(f"Indexed {len(self.video_files)} video files")

    def _extract_asana_name(self, filename: str) -> str:
        """Extract the Asana name from a video filename.
        
        Examples:
            - "Bhujangasana_h264_med" -> "Bhujangasana"
            - "SuryaNamaskar_720p" -> "SuryaNamaskar"
        """
        # Split by underscore, dash, or space and take the first meaningful part
        parts = re.split(r'[_\-\s]', filename)
        return parts[0] if parts else filename

    def find_matching_videos(self, text: str) -> List[Dict[str, Any]]:
        """Find all video files that match Asana names mentioned in the text."""
        matches = []
        text_lower = text.lower()
        matched_asanas = set()  # Avoid duplicates

        # Method 1: Match against indexed video filenames
        for asana_name, video_path in self.video_files.items():
            if asana_name in text_lower and asana_name not in matched_asanas:
                matches.append({
                    "asana_name": asana_name.capitalize(),
                    "video_path": str(video_path),
                    "video_filename": video_path.name,
                    "video_url": f"videos/{video_path.name}"
                })
                matched_asanas.add(asana_name)
                logger.info(f"Found matching video for '{asana_name}': {video_path.name}")

        # Method 2: Also check known asana names against video files
        for known_asana in self.KNOWN_ASANAS:
            if known_asana in text_lower and known_asana not in matched_asanas:
                # Check if we have a video for this asana
                for video_asana, video_path in self.video_files.items():
                    if known_asana in video_asana or video_asana in known_asana:
                        if video_asana not in matched_asanas:
                            matches.append({
                                "asana_name": video_asana.capitalize(),
                                "video_path": str(video_path),
                                "video_filename": video_path.name,
                                "video_url": f"videos/{video_path.name}"
                            })
                            matched_asanas.add(video_asana)

        return matches

    def get_available_asanas(self) -> List[str]:
        """Get list of all available Asana names with videos."""
        return [name.capitalize() for name in self.video_files.keys()]


class YogaBharatiOrchestrator:
    """
    Orchestrator for yoga class generation with video embeddings.
    
    Workflow:
    1. Receive user query
    2. Call RAG service (handles all AI processing)
    3. Parse response for Asana names
    4. Match Asanas with local videos
    5. Return formatted response with video links
    """

    def __init__(self):
        """Initialize the orchestrator."""
        # Initialize video matcher for local video files
        self.video_matcher = VideoMatcher()
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("YogaBharatiOrchestrator initialized")
        logger.info(f"Available asanas with videos: {self.video_matcher.get_available_asanas()}")

    def _format_response_with_videos(
        self, 
        base_response: str, 
        video_matches: List[Dict[str, Any]]
    ) -> str:
        """Format the response with embedded video references."""
        if not video_matches:
            return base_response

        # Add video section to response
        video_section = "\n\n---\n## 🎥 Video Demonstrations\n\n"
        video_section += "The following video demonstrations are available for the asanas mentioned:\n\n"

        for video in video_matches:
            video_section += f"### {video['asana_name']}\n"
            video_section += f"📹 **Video File:** `{video['video_filename']}`\n"
            video_section += f"📂 **Path:** `{video['video_url']}`\n\n"

        video_section += "---\n"
        video_section += "*Videos from YogaBharati local library*\n"

        return base_response + video_section

    def _generate_video_embed_html(self, video: Dict[str, Any]) -> str:
        """Generate HTML embed code for a video."""
        return f'''<div class="yoga-video-container">
    <h4>{video["asana_name"]} Demonstration</h4>
    <video width="640" height="360" controls>
        <source src="{video["video_url"]}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
</div>'''

    def _generate_video_embed_markdown(self, video: Dict[str, Any]) -> str:
        """Generate Markdown embed reference for a video."""
        return f'''### 🎬 {video["asana_name"]} Video Demonstration
**Video File:** [{video["video_filename"]}]({video["video_url"]})
'''

    def process_user_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query: call RAG service, match videos, format response.
        
        Args:
            user_query: The user's yoga-related query
            
        Returns:
            Dict containing:
            - query: Original query
            - response: Formatted response with video links
            - rag_output: Raw RAG service response
            - matching_videos: List of matched video info
            - video_embeds: HTML/Markdown embeds for videos
        """
        logger.info(f"Processing user query: {user_query}")

        # Step 1: Call the RAG service
        logger.info("Calling RAG service...")
        rag_result = query_rag_system(query_text=user_query, top_k=5)
        
        # Check for errors
        if rag_result.get("error"):
            logger.error(f"RAG service error: {rag_result['error']}")
            return {
                "query": user_query,
                "response": f"Error from RAG service: {rag_result['error']}",
                "rag_output": rag_result,
                "matching_videos": [],
                "video_embeds": [],
                "error": rag_result['error']
            }

        # Step 2: Extract the response text
        rag_response = rag_result.get("response", "") or rag_result.get("context", "")
        logger.info(f"RAG response received ({len(rag_response)} chars)")

        # Step 3: Find matching videos in the response
        matching_videos = self.video_matcher.find_matching_videos(rag_response)
        logger.info(f"Found {len(matching_videos)} matching videos")

        # Step 4: Generate video embeds
        video_embeds = []
        for video in matching_videos:
            video_embeds.append({
                "asana": video["asana_name"],
                "video_file": video["video_filename"],
                "embed_html": self._generate_video_embed_html(video),
                "embed_markdown": self._generate_video_embed_markdown(video),
            })

        # Step 5: Format the final response with video links
        final_response = self._format_response_with_videos(rag_response, matching_videos)

        # Store in conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": final_response})

        return {
            "query": user_query,
            "response": final_response,
            "rag_output": rag_result,
            "matching_videos": matching_videos,
            "video_embeds": video_embeds,
        }

    def process_user_input(self, user_input: str) -> str:
        """
        Simple interface - process input and return just the response string.
        
        Args:
            user_input: User's query
            
        Returns:
            Formatted response string with video links
        """
        result = self.process_user_query(user_input)
        return result["response"]

    def chat(self, user_input: str) -> str:
        """Alias for process_user_input."""
        return self.process_user_input(user_input)

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")

    def get_available_video_asanas(self) -> List[str]:
        """Get list of all available Asana names with videos."""
        return self.video_matcher.get_available_asanas()


# Convenience function
def query_with_video_matching(user_query: str) -> Dict[str, Any]:
    """
    Convenience function to query RAG and match videos.
    
    Args:
        user_query: The user's yoga-related query
        
    Returns:
        Dict containing response, RAG output, and video matches
    """
    orchestrator = YogaBharatiOrchestrator()
    return orchestrator.process_user_query(user_query)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("YogaBharati Orchestrator - RAG + Video Matching")
    print("=" * 60)
    
    try:
        orchestrator = YogaBharatiOrchestrator()
        
        # Show available asanas with videos
        available_asanas = orchestrator.get_available_video_asanas()
        print(f"\n📹 Available asanas with videos: {available_asanas}")
        
        # Example query
        example_query = "Create a beginner yoga class with Bhujangasana and basic stretches"
        print(f"\n📝 Example Query: {example_query}")
        print("-" * 40)
        
        result = orchestrator.process_user_query(example_query)
        
        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")
        else:
            print("\n✅ Response:")
            print(result["response"])
            
            if result["matching_videos"]:
                print("\n\n🎬 Matching Videos Found:")
                for video in result["matching_videos"]:
                    print(f"   - {video['asana_name']}: {video['video_filename']}")
            else:
                print("\n⚠️ No matching videos found in local library")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()