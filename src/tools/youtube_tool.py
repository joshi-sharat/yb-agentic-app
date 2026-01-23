"""
YouTube search tool for finding relevant YogaBharati videos.
"""
import logging
from typing import Optional
from youtube_search import YoutubeSearch

logger = logging.getLogger(__name__)


def search_yogabharati_videos(query: str, max_results: int = 10) -> list[dict]:
    """
    Search YogaBharati YouTube channel for videos matching the query.

    Args:
        query: Search query for yoga class videos
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of video information with titles, descriptions, and links
    """
    try:
        logger.info(f"Searching YogaBharati channel for: {query}")

        # Search videos on YouTube
        videos_search = YoutubeSearch(query, limit=max_results)
        results = videos_search.result()

        videos = []
        for video in results.get("result", []):
            video_info = {
                "title": video.get("title", ""),
                "description": video.get("description", ""),
                "link": video.get("link", ""),
                "channel": video.get("channel", {}).get("name", ""),
                "duration": video.get("duration", ""),
                "published_at": video.get("publishedTime", ""),
                "thumbnails": video.get("thumbnails", []),
            }
            if "YogaBharati" in video_info.get("channel", "") or "yogabharati" in query.lower():
                videos.append(video_info)

        logger.info(f"Found {len(videos)} YogaBharati videos for query: {query}")
        return videos

    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        return []


def get_channel_videos(
    channel_name: str = "YogaBharati", max_results: int = 10
) -> list[dict]:
    """
    Get recent videos from the YogaBharati channel.

    Args:
        channel_name: Name of the channel to search
        max_results: Maximum number of results to return

    Returns:
        List of recent videos from the channel
    """
    try:
        logger.info(f"Fetching videos from {channel_name} channel")

        # Search for videos from the specified channel
        videos_search = YoutubeSearch(f"{channel_name} yoga", limit=max_results)
        video_results = videos_search.result()

        videos = []
        for video in video_results.get("result", []):
            # Filter to only include videos from the target channel
            if channel_name.lower() in video.get("channel", {}).get("name", "").lower():
                video_info = {
                    "title": video.get("title", ""),
                    "description": video.get("description", ""),
                    "link": video.get("link", ""),
                    "channel": video.get("channel", {}).get("name", ""),
                    "duration": video.get("duration", ""),
                    "published_at": video.get("publishedTime", ""),
                }
                videos.append(video_info)

        logger.info(f"Retrieved {len(videos)} videos from {channel_name}")
        return videos

    except Exception as e:
        logger.error(f"Error fetching channel videos: {str(e)}")
        return []
