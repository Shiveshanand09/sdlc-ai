import os
import pytest
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from sdlc_ai_project.agents.knowledge import KnowledgeBaseAgent
from sdlc_ai_project.utils import collect_task_outputs
from sdlc_ai_project.tools import code_research_tool
from sdlc_ai_project.llms import llama_llm
# Load environment variables
load_dotenv()

gemini_api_key =  os.getenv("GEMINI_API_KEY")

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")

@pytest.fixture
def knowledge_agent():
    return KnowledgeBaseAgent(
        user_requirements="Build a todo app with user authentication",
        project_context="A simple web application for task management",
        llm=llama_llm,
    )

def test_knowledge_agent(knowledge_agent):
    outputs = collect_task_outputs(knowledge_agent)
    assert outputs is not None
    assert isinstance(outputs, dict)
    assert len(outputs)>0

    print("\nKnowledge Result:")
    print(outputs)

if __name__ == "__main__":
    pytest.main([__file__])
