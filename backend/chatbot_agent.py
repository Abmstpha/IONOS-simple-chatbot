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
    "1. ALWAYS start by using the search_documents tool to search your loaded documents first\n"
    "2. If the documents don't contain enough relevant information, then use web_search as a fallback\n"
    "3. Combine information from both sources when available\n"
    "4. Provide comprehensive, accurate answers based on the retrieved information\n\n"
    "When searching documents, look for any relevant information that could help answer the user's question. "
    "Only use web_search if the documents don't provide sufficient information.\n\n"
    "Always be helpful and provide detailed responses based on the information you find."
)


@tool
def search_documents(query: str, config: RunnableConfig) -> str:
    """
    Search your loaded documents for relevant information. Use this first before web search.
    """
    logger.info(f"Searching scrapped web page for: {query}")
    
    # Check if retriever is available
    if "retriever" not in config.get("configurable", {}):
        logger.warning("No retriever available in config")
        return "No documents have been loaded yet. Please initialize RAG first by providing a URL."
    
    try:
        retriever = config["configurable"]["retriever"]
        chunks = retriever.invoke(query, k=8)
        logger.info(f"Found {len(chunks)} document chunks")
        
        if not chunks:
            return "No relevant information found in the loaded documents."
        
        # Combine all chunk contents
        content = "\n\n".join(chunk.page_content for chunk in chunks)
        return f"Found {len(chunks)} relevant document chunks:\n\n{content}"
        
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
        return f"Found {len(chunks)} web search results:\n\n{content}"
        
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
    return create_react_agent(model=llm, prompt=_prompt, tools=[search_documents, web_search], state_schema=AgentStatePydantic)
