"""
Video indexer for extracting metadata from local video files.
"""
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def safe_parse_fps(fps_string: str) -> float:
    """
    Safely parse frame rate string like '30/1' or '29.97'.
    
    Args:
        fps_string: Frame rate string from ffprobe
        
    Returns:
        Float frame rate value, or 0.0 if parsing fails
    """
    try:
        if '/' in fps_string:
            num, den = fps_string.split('/')
            denominator = float(den)
            if denominator == 0:
                return 0.0
            return float(num) / denominator
        return float(fps_string)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


class VideoIndexer:
    """Index and extract metadata from video files in a folder."""
    
    SUPPORTED_FORMATS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv')
    
    def __init__(self, video_folder: str):
        """
        Initialize the video indexer.
        
        Args:
            video_folder: Path to the folder containing video files
        """
        self.video_folder = video_folder
        self.videos = self.index_videos()

    def index_videos(self) -> list[str]:
        """
        Index all video files in the specified folder.
        
        Returns:
            List of video filenames found
        """
        video_files = []
        
        if not os.path.isdir(self.video_folder):
            logger.warning(f"Video folder does not exist: {self.video_folder}")
            return video_files
            
        for filename in os.listdir(self.video_folder):
            if filename.lower().endswith(self.SUPPORTED_FORMATS):
                video_files.append(filename)
                metadata = self.get_video_metadata(filename)
                logger.info(f"Indexed video: {filename} - Duration: {metadata.get('duration', 'N/A')}s")
                
        logger.info(f"Total videos indexed: {len(video_files)}")
        return video_files

    def get_video_metadata(self, filename: str) -> dict:
        """
        Extract comprehensive metadata from video files using ffprobe.
        
        Args:
            filename: Name of the video file
            
        Returns:
            Dictionary containing video metadata
        """
        file_path = os.path.join(self.video_folder, filename)
        
        metadata = {
            'name': filename,
            'path': file_path,
            'size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        }

        try:
            # Use ffprobe to extract metadata
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                probe_data = json.loads(result.stdout)

                # Extract format metadata
                if 'format' in probe_data:
                    format_data = probe_data['format']
                    metadata['duration'] = float(format_data.get('duration', 0))
                    metadata['bit_rate'] = int(format_data.get('bit_rate', 0))

                    # Extract tags
                    if 'tags' in format_data:
                        tags = format_data['tags']
                        metadata['title'] = tags.get('title', 'N/A')
                        metadata['subtitle'] = tags.get('subtitle', 'N/A')
                        metadata['artist'] = tags.get('artist', 'N/A')
                        metadata['album'] = tags.get('album', 'N/A')
                        metadata['comment'] = tags.get('comment', 'N/A')
                        metadata['date'] = tags.get('date', 'N/A')
                        metadata['genre'] = tags.get('genre', 'N/A')
                        metadata['other_tags'] = tags.get('tags', 'N/A')

                # Extract stream metadata (video/audio info)
                if 'streams' in probe_data:
                    for stream in probe_data['streams']:
                        if stream.get('codec_type') == 'video':
                            metadata['video'] = {
                                'codec': stream.get('codec_name', 'N/A'),
                                'width': stream.get('width', 'N/A'),
                                'height': stream.get('height', 'N/A'),
                                'fps': safe_parse_fps(stream.get('r_frame_rate', '0/1')),
                            }
                        elif stream.get('codec_type') == 'audio':
                            metadata['audio'] = {
                                'codec': stream.get('codec_name', 'N/A'),
                                'sample_rate': stream.get('sample_rate', 'N/A'),
                                'channels': stream.get('channels', 'N/A'),
                            }

                # Log extracted metadata
                self._log_metadata(filename, metadata)

        except FileNotFoundError:
            logger.warning("ffprobe not found. Install FFmpeg to extract full metadata.")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout extracting metadata for {filename}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse ffprobe output for {filename}: {str(e)}")
        except Exception as e:
            logger.warning(f"Could not extract full metadata for {filename}: {str(e)}")

        return metadata
    
    def _log_metadata(self, filename: str, metadata: dict) -> None:
        """Log extracted metadata in a formatted way."""
        logger.debug(f"\n=== Metadata for {filename} ===")
        logger.debug(f"Title: {metadata.get('title', 'N/A')}")
        logger.debug(f"Subtitle: {metadata.get('subtitle', 'N/A')}")
        logger.debug(f"Size: {metadata.get('size', 0)} bytes")
        logger.debug(f"Duration: {metadata.get('duration', 'N/A')} seconds")
        logger.debug(f"Bit Rate: {metadata.get('bit_rate', 'N/A')} bps")
        logger.debug(f"Artist: {metadata.get('artist', 'N/A')}")
        logger.debug(f"Album: {metadata.get('album', 'N/A')}")
        logger.debug(f"Comment: {metadata.get('comment', 'N/A')}")
        logger.debug(f"Date: {metadata.get('date', 'N/A')}")
        logger.debug(f"Genre: {metadata.get('genre', 'N/A')}")
        logger.debug(f"Other Tags: {metadata.get('other_tags', 'N/A')}")

        if 'video' in metadata:
            v = metadata['video']
            logger.debug(f"Video: {v['codec']} ({v['width']}x{v['height']}) @ {v['fps']} fps")

        if 'audio' in metadata:
            a = metadata['audio']
            logger.debug(f"Audio: {a['codec']} ({a['channels']} channels, {a['sample_rate']} Hz)")

    def get_videos_info(self) -> list[dict]:
        """
        Get metadata for all indexed videos.
        
        Returns:
            List of metadata dictionaries for all videos
        """
        return [self.get_video_metadata(video) for video in self.videos]
    
    def search_videos(self, query: str) -> list[dict]:
        """
        Search videos by keyword in metadata.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching video metadata dictionaries
        """
        query_lower = query.lower()
        results = []
        
        for video in self.videos:
            metadata = self.get_video_metadata(video)
            
            # Search in relevant fields
            searchable = " ".join([
                str(metadata.get('title', '')),
                str(metadata.get('subtitle', '')),
                str(metadata.get('comment', '')),
                str(metadata.get('genre', '')),
                str(metadata.get('name', ''))
            ]).lower()
            
            if query_lower in searchable:
                results.append(metadata)
        
        return results


if __name__ == "__main__":
    import sys
    
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python video_indexer.py <video_folder>")
        sys.exit(1)

    video_folder = sys.argv[1]

    if not os.path.isdir(video_folder):
        print(f"Error: '{video_folder}' is not a valid directory")
        sys.exit(1)

    indexer = VideoIndexer(video_folder)
    videos_info = indexer.get_videos_info()

    print(f"\nFound {len(videos_info)} video(s):")
    for video in videos_info:
        print(f"  - {video['name']} ({video.get('size', 0)} bytes)")
