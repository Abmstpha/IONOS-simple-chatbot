import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from pydantic import SecretStr

load_dotenv()

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("chatbot-server")

_prompt: str = (
    "You are an expert AI assistant with access to both document search and web search capabilities.\n\n"
    "IMPORTANT WORKFLOW:\n"
    "1. If documents are available, start by using the search_documents tool to search your loaded documents\n"
    "2. If no documents are available OR if documents don't contain enough information, use web_search\n"
    "3. After gathering information, provide a clear, concise answer to the user's question\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- After using any tools, you MUST provide a natural language response to the user\n"
    "- Do NOT return tool calls or raw tool results\n"
    "- Always end your response with a helpful answer based on the information you found\n"
    "- If you use tools, analyze the results and provide a comprehensive response\n"
    "- If the search_documents tool returns 'No documents have been loaded yet', immediately use web_search instead\n\n"
    "SMART SOURCE ATTRIBUTION:\n"
    "- When using document search results, naturally mention 'based on the provided resources' or 'according to the loaded documents'\n"
    "- When using web search results, naturally mention 'according to recent information' or 'based on current sources'\n"
    "- Vary your language - don't use the same phrase every time\n"
    "- Be subtle and natural, not repetitive\n\n"
    "Always be helpful and provide clear, concise answers based on the retrieved information."
)


@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """
    Search your loaded documents for relevant information. Use this first before web search.
    """
    logger.info(f"Searching scrapped web page for: {query}")
    
    # Check if retriever is available
    if "retriever" not in config.get("configurable", {}) or config["configurable"]["retriever"] is None:
        logger.info("No retriever available - skipping document search")
        return "No documents have been loaded yet. Please use web_search to find information."
    
    try:
        retriever = config["configurable"]["retriever"]
        chunks = retriever.invoke(query, k=8)
        logger.info(f"Found {len(chunks)} document chunks")
        
        if not chunks:
            return "No relevant information found in the loaded documents."
        
        # Combine all chunk contents
        content = "\n\n".join(chunk.page_content for chunk in chunks)
        return f"Document search results ({len(chunks)} chunks): {content}"
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return f"Error searching documents: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Search the web for additional information. Use this as a fallback when documents don't have enough information.
    """
    logger.info(f"Searching web for: {query}")
    try:
        retriever = TavilySearchAPIRetriever(k=8)
        chunks = retriever.invoke(query)
        logger.info(f"Found {len(chunks)} web search results")
        
        if not chunks:
            return "No relevant information found on the web."
        
        # Combine all chunk contents
        content = "\n\n".join(chunk.page_content for chunk in chunks)
        return f"Web search results ({len(chunks)} results): {content}"
        
    except Exception as e:
        logger.error(f"Error in web search: {e}")
        return f"Error searching the web: {str(e)}"


def create_chatbot_agent(model_name: str) -> CompiledStateGraph:
    llm = ChatOpenAI(
        model=model_name,
        base_url="https://openai.inference.de-txl.ionos.com/v1",
        api_key=SecretStr(os.getenv("IONOS_API_KEY", "")),
        temperature=0,
        max_tokens=1024,
    )
    
    # Create the ReAct agent with proper configuration
    agent = create_react_agent(
        model=llm, 
        prompt=_prompt, 
        tools=[search_documents, web_search], 
        state_schema=AgentStatePydantic
    )
    
    return agent
