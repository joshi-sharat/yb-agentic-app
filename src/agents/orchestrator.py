"""
YogaBharati Agentic Orchestrator - Main Agent Implementation
Orchestrates uses input, searches YouTube, and integrates with local RAG system.
"""
import json
import logging
from typing import Any, Optional

from agent_framework_azure_ai import AzureAIAgentClient
from agent_framework_azure_ai._chat_client import Agent, ChatMessage, MessageRole
from openai import AzureOpenAI

from config.settings import settings
from src.tools import (
    search_yogabharati_videos,
    query_rag_system,
    search_yoga_class_details,
)

logger = logging.getLogger(__name__)


class YogaBharatiOrchestrator:
    """Orchestrator agent for yoga class creation with appropriate video embeddings from local library for practices in the class generated."""

    def __init__(self):
        """Initialize the orchestrator agent."""
        # Validate settings before initializing
        self._validate_settings()
        
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.API_VERSION,
            azure_endpoint=settings.API_ENDPOINT,
        )
        self.agent = self._create_agent()
        self.conversation_history: list[ChatMessage] = []
        
        logger.info("YogaBharatiOrchestrator initialized successfully")

    def _validate_settings(self) -> None:
        """Validate required settings are configured."""
        if not settings.AZURE_OPENAI_API_KEY:
            raise ValueError(
                "Azure OpenAI API key not configured. "
                "Set the AZURE_OPENAI_API_KEY environment variable."
            )
        
        if not settings.API_ENDPOINT:
            raise ValueError(
                "API endpoint not configured. "
                "Set the API_ENDPOINT environment variable."
            )
        
        logger.debug("Settings validated successfully")

    def _create_agent(self) -> Agent:
        """Create and configure the agent with tools."""
        agent = Agent(
            name="YogaBharati Orchestrator",
            instructions="""You are an expert yoga expert and instructor of YogaBharati. Use your expertise in Yoga to generate appropriate Yoga classes.

Your role is to:
1. Understand the user's yoga practice needs and preferences
2. Search the YogaBharati YouTube channel for relevant video snippets
3. Query the local RAG system for detailed yoga class information
4. Recommend personalized yoga classes with corresponding video links

When a user asks about yoga classes:
- First, understand their experience level and goals
- Query the RAG system to get detailed class information
- Search local video library i.e videos folder for relevant YogaBharati videos
- Combine both results to provide comprehensive recommendations
- Format results with video links and class details clearly

Be encouraging, knowledgeable, and personalized in your responses.
Always cite the video sources and class materials you recommend.""",
            client=self.client,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "query_rag_system",
                        "description": "Query the RAG system for detailed yoga class generation based on user input",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query_text": {
                                    "type": "string",
                                    "description": "give a standard YogaBharati Yoga class"
                                    " Make sure that it is max 75 minutes "
                                    "Has an Opening Prayer and Ending Relaxation practices "
                                    "Name each practice, the duration of it and the timeline "
                                    "Be short and Crisp, DONT Repeat",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return (default: 5)",
                                    "default": 5,
                                },
                            },
                            "required": ["query_text"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_yogabharati_videos",
                        "description": "Search YogaBharati local library for videos matching a query",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query for yoga videos (e.g., 'Bhujangasana', 'SuryaNamaskar')",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of videos to return (default: 10)",
                                    "default": 10,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_yoga_class_details",
                        "description": "Search for detailed yoga class information by class type",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "class_type": {
                                    "type": "string",
                                    "description": "Type of yoga class (e.g., 'beginner', 'advanced', 'meditation')",
                                },
                            },
                            "required": ["class_type"],
                        },
                    },
                },
            ],
        )
        return agent

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Execute a tool and return results."""
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        try:
            if tool_name == "search_yogabharati_videos":
                return search_yogabharati_videos(
                    query=tool_args.get("query", ""),
                    max_results=tool_args.get("max_results", 10),
                )
            elif tool_name == "query_rag_system":
                return query_rag_system(
                    query_text=tool_args.get("query_text", ""),
                    top_k=tool_args.get("top_k", 5),
                )
            elif tool_name == "search_yoga_class_details":
                return search_yoga_class_details(
                    class_type=tool_args.get("class_type", "")
                )
            else:
                logger.error(f"Unknown tool: {tool_name}")
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": str(e)}

    def process_user_input(self, user_input: str) -> str:
        """
        Process user input and generate a response with recommendations.

        Args:
            user_input: User's query about yoga classes

        Returns:
            Agent's response with video suggestions and class recommendations
        """
        logger.info(f"Processing user input: {user_input}")

        # Add user message to history
        self.conversation_history.append(
            ChatMessage(role=MessageRole.USER, content=user_input)
        )

        # Call the agent
        messages = self.conversation_history.copy()

        # Keep making requests until we get a final response (no more tool calls)
        max_iterations = settings.MAX_ITERATIONS
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}")

            # Get response from agent
            response = self.agent.invoke(messages)
            logger.info(f"Agent response: {response}")

            # Parse response
            if isinstance(response, str):
                # Final text response
                self.conversation_history.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=response)
                )
                return response

            # Check if there are tool calls to process
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No tool calls, return the response content
                response_content = self._extract_response_content(response)
                if response_content:
                    self.conversation_history.append(
                        ChatMessage(role=MessageRole.ASSISTANT, content=response_content)
                    )
                    return response_content

            # Execute tool calls and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_result = self._execute_tool(tool_call["name"], tool_call["args"])
                tool_results.append(
                    {
                        "tool_call_id": tool_call.get("id", ""),
                        "tool_name": tool_call["name"],
                        "result": json.dumps(tool_result, default=str),
                    }
                )

            # Add tool results to messages
            messages = self._add_tool_results_to_messages(messages, tool_results)

        logger.warning("Max iterations reached without final response")
        return "I apologize, but I encountered an issue processing your request. Please try again."

    def _extract_tool_calls(self, response: Any) -> list[dict]:
        """Extract tool calls from agent response."""
        try:
            if hasattr(response, "tool_calls"):
                return [
                    {
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    }
                    for tc in response.tool_calls
                ]
            return []
        except Exception as e:
            logger.error(f"Error extracting tool calls: {str(e)}")
            return []

    def _extract_response_content(self, response: Any) -> str:
        """Extract text content from agent response."""
        try:
            if hasattr(response, "content"):
                return response.content
            elif isinstance(response, str):
                return response
            elif hasattr(response, "message"):
                return response.message
            return ""
        except Exception as e:
            logger.error(f"Error extracting response content: {str(e)}")
            return ""

    def _add_tool_results_to_messages(
        self, messages: list, tool_results: list[dict]
    ) -> list:
        """Add tool results to the message history."""
        for result in tool_results:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=result["result"],
                    tool_call_id=result["tool_call_id"],
                )
            )
        return messages

    def get_conversation_history(self) -> list[dict]:
        """Get the conversation history."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.conversation_history
        ]

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")
