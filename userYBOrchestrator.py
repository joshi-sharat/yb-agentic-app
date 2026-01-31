from src.utils import setup_logging
from src.agents.orchestrator import YogaBharatiOrchestrator


# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


orchestrator = YogaBharatiOrchestrator()

# Process query
result = orchestrator.process_user_query("Create a yoga class with Bhujangasana")

print(result["response"])        # RAG response + video links
print(result["matching_videos"]) # List of matched videos