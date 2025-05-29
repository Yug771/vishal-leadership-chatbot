import streamlit as st
import asyncio
import os

from llama_index.core import ChatPromptTemplate # QueryBundle removed as not directly used
from llama_index.core.prompts import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

# Configuration values that are not secret can remain here or be part of secrets.toml
# For now, let's assume these could also be in secrets.toml if preferred, but are less sensitive than API keys.
DEFAULT_LLAMA_CLOUD_PROJECT_NAME = "Default"
DEFAULT_LLAMA_CLOUD_INDEX_NAME = "vishal-pdf-parsing"

# Page Configuration
st.set_page_config(page_title="Leadership Insights with Prof. Gupta", layout="wide", page_icon="💡")

# --- Helper Functions ---

@st.cache_resource
def load_agent(openai_api_key_param, llama_cloud_api_key_param, llama_cloud_project_name_param, llama_cloud_index_name_param, llama_cloud_organization_id_param):
    """
    Loads and initializes the LlamaIndex agent.
    Uses st.cache_resource to cache the agent across reruns.
    Returns None if essential keys are missing or agent setup fails.
    """
    # Essential keys check is now primarily outside, before calling this function.
    # This function assumes valid keys are passed, but os.environ needs to be set.
    os.environ["OPENAI_API_KEY"] = openai_api_key_param

    try:
        index = LlamaCloudIndex(
            name=llama_cloud_index_name_param,
            project_name=llama_cloud_project_name_param,
            organization_id=llama_cloud_organization_id_param,
            api_key=llama_cloud_api_key_param,
        )

        retriever = index.as_retriever(
            search_type="default",
            search_kwargs={
                "similarity_top_k": 30,
                "node_types": ["chunk"],
                "rerank": True,
                "rerank_top_n": 6,
                "filter_mode": "accurate",
                "multimodal": False,
                "filter_condition": {"operator": "AND", "filters": []}
            }
        )

        system_prompt_content = """I am Professor Vishal Gupta from IIM Ahmedabad. I will help you with your questions about my Leadership Skills course. My responses will be comprehensive, accurate, and relevant, based *only* on my course transcripts. Adhere to the following guidelines when I answer:

1. **Exclusive Use of My Course Content**: I will ONLY use information from my course transcripts. I will not use any external knowledge or other sources.
2. **Accurate Reference**: I will always include the relevant week and topic title(s) in my answers, formatting it as: [Week X: Topic Title].
3. **Handling Unanswerable Questions**: If your question cannot be answered using my course transcripts, I will state this clearly.
4. **Strict Non-Inference Policy**: I will not infer information not explicitly stated in my course content.
5. **Structured and Clear Responses**: My responses will be well-structured, and I will quote directly from my transcript when appropriate.
6. **Direct and Guiding Tone**: I will respond directly to you, offering guidance and insights based on my course material.
7. **Comprehensive Answers**: I will provide thorough answers, elaborating on key points and connecting ideas from different parts of my course when relevant.
8. **Consistency**: I will maintain consistency in style and adherence to these guidelines throughout our conversation.

Remember, accuracy and relevance to my course content are paramount in my responses."""

        message_templates = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_content),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query using chain-of-thought reasoning: {query_str}\n"
                ),
            ),
        ]
        custom_prompt = ChatPromptTemplate(message_templates=message_templates)

        llm = OpenAI(model="gpt-4o", temperature=0.1)

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm,
            prompt=custom_prompt,
        )

        query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="course_material_search_tool",
            description="Tool for me (Professor Vishal Gupta) to search and retrieve information from my Leadership Skills course transcripts to answer student questions.",
            return_direct=True,
        )

        agent = FunctionAgent(
            tools=[query_engine_tool],
            llm=llm,
            system_prompt=system_prompt_content
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing the agent: {e}. Please check your LlamaCloud configuration and API keys in secrets.toml.")
        return None

async def get_agent_response(agent_instance, user_query):
    """
    Asynchronously gets the agent's response.
    """
    agent_output = await agent_instance.run(user_query)
    if hasattr(agent_output, 'response') and hasattr(agent_output.response, 'content'):
        return str(agent_output.response.content)
    return str(agent_output)

# --- Streamlit UI ---

st.title("💡 Leadership Insights with Prof. Gupta") # Catchier Title
st.markdown("Welcome! I'm Professor Vishal Gupta. How can I guide you through the Leadership Skills course today?")

# Sidebar for Configuration Information
# with st.sidebar:
#     st.header("Configuration")
#     st.warning("Please ensure your API keys (OpenAI, LlamaCloud) and LlamaCloud details (Project Name, Index Name, Organization ID) are correctly set up in `.streamlit/secrets.toml`.", icon="🔒")
#     st.markdown("Example `secrets.toml` structure:")
#     st.code("""
# OPENAI_API_KEY = "sk-..."
# LLAMA_CLOUD_API_KEY = "llx-..."
# LLAMA_CLOUD_PROJECT_NAME = "Default"
# LLAMA_CLOUD_INDEX_NAME = "your-index-name"
# LLAMA_CLOUD_ORGANIZATION_ID = "your-org-id"
# """, language="toml")

# Load configurations from secrets
agent = None
keys_loaded_successfully = False
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    llama_cloud_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
    llama_cloud_project_name = st.secrets.get("LLAMA_CLOUD_PROJECT_NAME", DEFAULT_LLAMA_CLOUD_PROJECT_NAME)
    llama_cloud_index_name = st.secrets.get("LLAMA_CLOUD_INDEX_NAME", DEFAULT_LLAMA_CLOUD_INDEX_NAME)
    llama_cloud_organization_id = st.secrets["LLAMA_CLOUD_ORGANIZATION_ID"]
    keys_loaded_successfully = True
except KeyError as e:
    st.error(f"Missing configuration in `.streamlit/secrets.toml`: {e}. Please create or update the file.", icon="🚨")
except FileNotFoundError:
    st.error("`.streamlit/secrets.toml` file not found. Please create it with your API keys and configuration.", icon="🚨")

if keys_loaded_successfully:
    # Attempt to load the agent only if keys were found
    # The spinner here will cover the agent loading process
    with st.spinner("Connecting with Professor Gupta's knowledge base..."):
        agent = load_agent(
            openai_api_key,
            llama_cloud_api_key,
            llama_cloud_project_name,
            llama_cloud_index_name,
            llama_cloud_organization_id
        )
    if agent is None and keys_loaded_successfully:
         # This means keys were found, but load_agent itself failed (e.g. bad key, network issue)
         # load_agent will have already displayed an error via st.error
         pass # Error is handled within load_agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm Professor Vishal Gupta. Ask me anything about the Leadership Skills course."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Your question for Professor Gupta..."):
    if not keys_loaded_successfully:
        st.error("Application is not configured. Please check `.streamlit/secrets.toml`.", icon="🚨")
    elif agent is None:
        st.error("Unable to connect to Professor Gupta at the moment. Please ensure your configuration in `.streamlit/secrets.toml` is correct and try again later.", icon="🚨")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Using a more engaging spinner text for query processing
            with st.spinner("Professor Gupta is thinking..."):
                try:
                    assistant_response = asyncio.run(get_agent_response(agent, user_input))
                    message_placeholder.markdown(assistant_response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    assistant_response = "My apologies, I seem to be having a bit of trouble recalling that information right now. Could you try rephrasing or asking again shortly?"
                    message_placeholder.markdown(assistant_response)
        
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

elif not keys_loaded_successfully:
    # This message shows if secrets weren't loaded on initial page load (before any input)
    st.info("Please set up your API keys in `.streamlit/secrets.toml` to begin your consultation with Professor Gupta.", icon="⚙️") 