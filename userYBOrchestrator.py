from src.agents.orchestrator import YogaBharatiOrchestrator

orchestrator = YogaBharatiOrchestrator()

# Process query
result = orchestrator.process_user_query("Create a yoga class with Bhujangasana")

print(result["response"])        # RAG response + video links
print(result["matching_videos"]) # List of matched videos