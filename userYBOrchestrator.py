
import logging
from src.utils import setup_logging
from src.agents.orchestrator import YogaBharatiOrchestrator


# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


orchestrator = YogaBharatiOrchestrator()

# Process query
print("=" * 60)
print("\n\nYogaBharati Class Generation Assistant !\n\n")
print("=" * 60)
uer_query = input("Please enter your request : ").strip()
print("invoking the generator, will be right back soon...\n\n")

result = orchestrator.process_user_query(uer_query)

print(result["response"])        # RAG response + video links
print(result["matching_videos"]) # List of matched videos