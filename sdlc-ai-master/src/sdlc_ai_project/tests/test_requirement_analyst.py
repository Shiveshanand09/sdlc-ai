import os
import pytest
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai import LLM
from sdlc_ai_project.agents.requirements import RequirementAnalyzer
from sdlc_ai_project.utils import collect_task_outputs
from sdlc_ai_project.tools import code_research_tool
from sdlc_ai_project.llms import gemini_llm
# Load environment variables
load_dotenv()

gemini_api_key =  os.getenv("GEMINI_API_KEY")

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")

@pytest.fixture
def requirement_analyst():
    return RequirementAnalyzer(
        user_requirements="Build a todo app with user authentication",
        project_context="A simple web application for task management",
        llm=gemini_llm,
    )


def test_capture_user_intent(requirement_analyst):
    """Test the agent's ability to capture user intent."""
    outputs = collect_task_outputs(requirement_analyst)
    assert outputs is not None
    assert isinstance(outputs, dict)
    assert len(outputs) > 0

    
    # Verify the output format
    # assert "## Original User Input" in result
    # assert "## Restructured Intent" in result
    # assert "## Key Components" in result
    # assert "## Technical Context" in result
    
    print("\nUser Intent Capture Result:")
    print(outputs)

# def test_analyze_requirements(requirement_analyst):
#     """Test the agent's ability to analyze requirements."""
#     # Create a task for analyzing requirements
#     task = Task(
#         description="""
#         You are a Senior Requirements Analyst AI working for a next-gen software engineering firm.
#         The user has provided a mix of business goals, feature ideas, and constraints in natural language.
        
#         Your responsibilities:
#         - Interpret ambiguous and vague statements
#         - Extract **explicit functional requirements**
#         - Detect **implicit requirements and constraints**
#         - Classify items into: Functional, Non-Functional, Assumptions, Constraints
#         - Identify any missing requirements or gaps

#         Output Format:
#         ## Functional Requirements
#         - ...
        
#         ## Non-Functional Requirements
#         - ...

#         ## Constraints
#         - ...

#         ## Assumptions
#         - ...

#         ## Gaps Identified
#         - ...
#         """,
#         agent=requirement_analyst,
#         expected_output="A structured requirements analysis in markdown with all five sections: Functional, Non-Functional, Constraints, Assumptions, and Gaps."
#     )
    
#     # Create a crew with just the requirement analyst
#     crew = Crew(
#         agents=[requirement_analyst],
#         tasks=[task]
#     )
    
#     # Test with a simple user input
#     result = crew.kickoff(inputs={
#         "user_requirements": "Build a todo app with user authentication and cloud sync"
#     })

#     result = result.raw
    
#     # Basic assertions
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     # # Verify the output format
#     # assert "## Functional Requirements" in result
#     # assert "## Non-Functional Requirements" in result
#     # assert "## Constraints" in result
#     # assert "## Assumptions" in result
#     # assert "## Gaps Identified" in result
    
#     print("\nRequirements Analysis Result:")
#     print(result)

# def test_end_to_end_analysis(requirement_analyst):
#     """Test the agent's ability to perform both intent capture and requirements analysis."""
#     # Create tasks for both intent capture and requirements analysis
#     intent_task = Task(
#         description="""
#         You are a User Intent Analysis AI working for a next-gen software engineering firm.
#         Your role is to restructure and enhance the user's initial input into a more comprehensive and clear statement of intent.
#         The user's input is: {user_intent}
#         Your responsibilities:
#         - Analyze the user's initial input
#         - Restructure the input into a more comprehensive statement
#         - Enhance the description with relevant technical context
#         - Maintain the core intent while adding necessary details
#         - Format the output for clear understanding

#         Output Format:
#         ## Original User Input
#         - ...

#         ## Restructured Intent
#         - A clear, comprehensive statement that captures the user's intent with added technical context and necessary details

#         ## Key Components
#         - Core functionality
#         - Technical requirements
#         - User experience goals
#         - Integration needs
#         - Performance expectations

#         ## Technical Context
#         - Relevant technologies
#         - Industry standards
#         - Best practices
#         - Common patterns
#         """,
#         agent=requirement_analyst,
#         expected_output="A restructured and enhanced version of the user's input that better captures their intent with added technical context and necessary details."
#     )
    
#     analysis_task = Task(
#         description="""
#         You are a Senior Requirements Analyst AI working for a next-gen software engineering firm.
#         The user has provided a mix of business goals, feature ideas, and constraints in natural language.
        
#         Your responsibilities:
#         - Interpret ambiguous and vague statements
#         - Extract **explicit functional requirements**
#         - Detect **implicit requirements and constraints**
#         - Classify items into: Functional, Non-Functional, Assumptions, Constraints
#         - Identify any missing requirements or gaps

#         Output Format:
#         ## Functional Requirements
#         - ...
        
#         ## Non-Functional Requirements
#         - ...

#         ## Constraints
#         - ...

#         ## Assumptions
#         - ...

#         ## Gaps Identified
#         - ...
#         """,
#         agent=requirement_analyst,
#         expected_output="A structured requirements analysis in markdown with all five sections: Functional, Non-Functional, Constraints, Assumptions, and Gaps."
#     )
    
#     # Create a crew with the requirement analyst and both tasks
#     crew = Crew(
#         agents=[requirement_analyst],
#         tasks=[intent_task, analysis_task]
#     )
    
#     # Test with a simple user input
#     result = crew.kickoff(inputs={
#         'user_intent': "Build a todo app with user authentication and cloud sync"
#     })

#     result = result.raw
    
#     # Basic assertions
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     # Verify the output format
#     # assert "## Original User Input" in result
#     # assert "## Restructured Intent" in result
#     # assert "## Key Components" in result
#     # assert "## Technical Context" in result
#     # assert "## Functional Requirements" in result
#     # assert "## Non-Functional Requirements" in result
#     # assert "## Constraints" in result
#     # assert "## Assumptions" in result
#     # assert "## Gaps Identified" in result
    
#     print("\nEnd-to-End Analysis Result:")
#     print(result)

if __name__ == "__main__":
    pytest.main([__file__]) 