import os
import subprocess
import json

class VideoIndexer:
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.videos = self.index_videos()

    def index_videos(self):
        video_files = []
        for filename in os.listdir(self.video_folder):
            if filename.endswith(('.mp4', '.mkv', '.avi')):  # Add more formats if needed
                video_files.append(filename)
            print(f"Indexed {filename} video file. " + str(self.get_video_metadata(filename)))
        return video_files

    def get_video_metadata(self, filename):
        """Extract comprehensive metadata from video files using ffprobe"""
        file_path = os.path.join(self.video_folder, filename)
        metadata = {
            'name': filename,
            'path': file_path,
            'size': os.path.getsize(file_path),
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
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
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
                                'fps': eval(stream.get('r_frame_rate', '0/1')),
                            }
                        elif stream.get('codec_type') == 'audio':
                            metadata['audio'] = {
                                'codec': stream.get('codec_name', 'N/A'),
                                'sample_rate': stream.get('sample_rate', 'N/A'),
                                'channels': stream.get('channels', 'N/A'),
                            }
                
                # Print extracted metadata
                print(f"\n=== Metadata for {filename} ===")
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"Subtitle: {metadata.get('subtitle', 'N/A')}")
                print(f"Size: {metadata['size']} bytes")
                print(f"Duration: {metadata.get('duration', 'N/A')} seconds")
                print(f"Bit Rate: {metadata.get('bit_rate', 'N/A')} bps")
                print(f"Artist: {metadata.get('artist', 'N/A')}")
                print(f"Album: {metadata.get('album', 'N/A')}")
                print(f"Comment: {metadata.get('comment', 'N/A')}")
                print(f"Date: {metadata.get('date', 'N/A')}")
                print(f"Genre: {metadata.get('genre', 'N/A')}")
                print(f"Other Tags: {metadata.get('other_tags', 'N/A')}")
                
                if 'video' in metadata:
                    v = metadata['video']
                    print(f"Video: {v['codec']} ({v['width']}x{v['height']}) @ {v['fps']} fps")
                
                if 'audio' in metadata:
                    a = metadata['audio']
                    print(f"Audio: {a['codec']} ({a['channels']} channels, {a['sample_rate']} Hz)")
                
        except FileNotFoundError:
            print(f"Warning: ffprobe not found. Install FFmpeg to extract full metadata.")
        except Exception as e:
            print(f"Warning: Could not extract full metadata for {filename}: {str(e)}")
        
        return metadata

    def get_videos_info(self):
        return [self.get_video_metadata(video) for video in self.videos]

if __name__ == "__main__":
    import sys
    
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
        print(f"  - {video['name']} ({video['size']} bytes)")