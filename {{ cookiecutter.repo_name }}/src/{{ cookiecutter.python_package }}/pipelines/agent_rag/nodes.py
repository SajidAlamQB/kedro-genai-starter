import logging
from typing import Any, Callable

import questionary
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableSerializable
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "No relevant context found"


def select_vector_store(
    vector_store_type: str,
    deeplake_vector_store_load=None,
    pinecone_vector_store_load=None,
) -> Any:
    """Selects the appropriate vector store based on the configuration.

    Args:
        vector_store_type: The type of vector store to use ("deeplake" or "pinecone")
        deeplake_vector_store_load: DeepLake vector store instance
        pinecone_vector_store_load: Pinecone vector store instance

    Returns:
        The selected vector store instance
    """
    if vector_store_type.lower() == "deeplake":
        if deeplake_vector_store_load is None:
            raise ValueError("DeepLake vector store is not initialized")
        return deeplake_vector_store_load
    elif vector_store_type.lower() == "pinecone":
        if pinecone_vector_store_load is None:
            print("WARNING: Pinecone vector store not available. Falling back to DeepLake.")
            if deeplake_vector_store_load is None:
                raise ValueError("DeepLake vector store (fallback) is not initialized")
            return deeplake_vector_store_load
        return pinecone_vector_store_load
    else:
        raise ValueError(f"Unsupported vector store type: {vector_store_type}")


def create_tools(
    vector_store: Any, embedding_function: Callable
) -> list[Callable]:
    """Creates a tool for retrieving context from the vector store based on user queries.

    Compatible with both DeepLake and Pinecone vector stores thanks to the adapter pattern
    implemented in the dataset classes.

    Args:
        vector_store: The vector store instance
        embedding_function: The function used for embedding user queries

    Returns:
        A list containing the retrieval tool
    """

    @tool
    def get_context_from_vector_store(user_question: str) -> str:
        """Returns the context found in vector store based on user question."""
        # Both DeepLake and Pinecone wrappers implement a compatible .search() method
        output = vector_store.search(
            embedding_data=user_question, embedding_function=embedding_function, k=1
        )["text"]

        return output[0] if output else FALLBACK_MESSAGE

    return [get_context_from_vector_store]


def init_llm(
    openai_llm: ChatOpenAI, tools: list[Callable]
) -> tuple[ChatOpenAI, Runnable]:
    """Initializes the LLM with provided tools.

    Args:
        openai_llm: The ChatOpenAI instance.
        tools: A list of tools for the agent.

    Returns:
        The initialized LLM and LLM bound with tools.
    """
    llm_with_tools = openai_llm.bind_tools(tools)
    return openai_llm, llm_with_tools


def create_chat_prompt(system_prompt: str) -> ChatPromptTemplate:
    """Creates a chat prompt template for the LLM.

    Args:
        system_prompt: The system prompt to guide responses.

    Returns:
        The formatted chat prompt template.
    """
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    return chat_prompt


def create_agent(
    llm_with_tools: Runnable, chat_prompt: ChatPromptTemplate
) -> RunnableSerializable:
    """Creates an AI agent using the provided LLM and chat prompt.

    Args:
        llm_with_tools: The LLM bound with tools.
        chat_prompt: The formatted chat prompt.

    Returns:
        RunnableSerializable: The AI agent instance.
    """
    agent: RunnableSerializable = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | chat_prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent


def create_agent_executor(
    agent: RunnableSerializable, tools: list[Callable]
) -> AgentExecutor:
    """Creates an agent executor to manage interactions with the AI agent.

    Args:
        agent: The AI agent instance.
        tools: Tools for the agent to use.

    Returns:
        The agent executor instance.
    """
    return AgentExecutor(
        agent=agent, tools=tools, verbose=False, return_intermediate_steps=True
    )


def invoke_agent(agent_executor: AgentExecutor, input_query: str) -> dict[str, Any]:
    """Invokes the agent executor with a given query.

    Args:
        agent_executor: The agent executor.
        input_query: The query to process.

    Returns:
        The agent's response.
    """
    return agent_executor.invoke({"input": input_query})


def invoke_llm(llm: ChatOpenAI, input_query: str) -> str:
    """Invokes the LLM directly with a given query.

    Args:
        llm: The ChatOpenAI instance.
        input_query: The query to process.

    Returns:
        The LLM's response.
    """
    return llm.invoke(input_query).text()


def user_interaction_loop(
    agent_executor: AgentExecutor,
    llm: ChatOpenAI,
    vector_store_type: str,
    user_query: str
) -> str:
    """Interactive loop to receive user input and process responses from the LLM and agent.

    Args:
        agent_executor: The agent executor.
        llm: The ChatOpenAI instance.
        vector_store_type: The type of vector store being used
        user_query: The query string passed through Kedro parameters.

    Returns:
        A formatted string containing all interactions.
    """
    res = []
    print(f"\n🔍 Using {vector_store_type.upper()} as the vector database backend\n")

    llm_response = invoke_llm(llm, user_query)
    agent_response = invoke_agent(agent_executor, user_query)

    input_res = f"### User Input: {user_query}\n"
    llm_res = f"### LLM Output:\n{llm_response}\n"
    agent_res = f"### Agent Output:\n{agent_response['output']}\n"
    agent_intermediate_steps = f"### Agent Intermediate Steps:\n```json\n{agent_response['intermediate_steps']}\n```\n"
    try:
        context = agent_response['intermediate_steps'][0][1]
    except IndexError:
        context = FALLBACK_MESSAGE
    retrieved_context = (
        f"### Retrieved Context:\n{context}\n"
    )
    vector_db_info = f"### Vector Database: {vector_store_type}\n"

    res.append(
        "\n".join(
            [
                input_res,
                llm_res,
                agent_res,
                retrieved_context,
                vector_db_info,
                agent_intermediate_steps,
            ]
        )
    )

    logger.info(input_res)
    logger.info(llm_res)
    logger.info(agent_res)

    return "\n\n".join(res)
