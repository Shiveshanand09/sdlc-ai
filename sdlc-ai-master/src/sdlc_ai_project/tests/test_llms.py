import os
import pytest
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from sdlc_ai_project.llms import gemini_llm, maidsr1_llm, llama_llm

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set in environment")

@pytest.fixture
def requirement_analyst_gemini():
    """Create a requirement analyst agent with Gemini LLM."""
    return Agent(
        role="Senior Requirements Analyst AI",
        goal="Translate user visions and vague inputs into well-structured and actionable development requirements.",
        backstory="""
        You are a world-class software analyst trained on millions of project specs and product briefs.
        Your expertise lies in transforming ambiguous user ideas into crystal-clear technical requirements,
        while anticipating edge cases, hidden constraints, and business logic gaps.
        You excel in NLP-driven requirement elicitation and business rule modeling.
        """,
        llm=gemini_llm
    )

@pytest.fixture
def requirement_analyst_maidsr1():
    """Create a requirement analyst agent with MAI-DS-R1 LLM."""
    return Agent(
        role="Senior Requirements Analyst AI",
        goal="Translate user visions and vague inputs into well-structured and actionable development requirements.",
        backstory="""
        You are a world-class software analyst trained on millions of project specs and product briefs.
        Your expertise lies in transforming ambiguous user ideas into crystal-clear technical requirements,
        while anticipating edge cases, hidden constraints, and business logic gaps.
        You excel in NLP-driven requirement elicitation and business rule modeling.
        """,
        llm=maidsr1_llm
    )

@pytest.fixture
def requirement_analyst_llama():
    """Create a requirement analyst agent with Llama LLM."""
    return Agent(
        role="Senior Requirements Analyst AI",
        goal="Translate user visions and vague inputs into well-structured and actionable development requirements.",
        backstory="""
        You are a world-class software analyst trained on millions of project specs and product briefs.
        Your expertise lies in transforming ambiguous user ideas into crystal-clear technical requirements,
        while anticipating edge cases, hidden constraints, and business logic gaps.
        You excel in NLP-driven requirement elicitation and business rule modeling.
        """,
        llm=llama_llm
    )

def test_gemini_llm(requirement_analyst_gemini):
    """Test the Gemini LLM."""
    # Create a task for capturing user intent
    task = Task(
        description="""
        You are a User Intent Analysis AI working for a next-gen software engineering firm.
        Your role is to restructure and enhance the user's initial input into a more comprehensive and clear statement of intent.
        
        Your responsibilities:
        - Analyze the user's initial input
        - Restructure the input into a more comprehensive statement
        - Enhance the description with relevant technical context
        - Maintain the core intent while adding necessary details
        - Format the output for clear understanding

        Output Format:
        ## Original User Input
        - ...

        ## Restructured Intent
        - A clear, comprehensive statement that captures the user's intent with added technical context and necessary details

        ## Key Components
        - Core functionality
        - Technical requirements
        - User experience goals
        - Integration needs
        - Performance expectations

        ## Technical Context
        - Relevant technologies
        - Industry standards
        - Best practices
        - Common patterns
        """,
        agent=requirement_analyst_gemini,
        expected_output="A restructured and enhanced version of the user's input that better captures their intent with added technical context and necessary details."
    )
    
    # Create a crew with just the requirement analyst
    crew = Crew(
        agents=[requirement_analyst_gemini],
        tasks=[task]
    )
    
    # Test with a simple user input
    result = crew.kickoff(inputs={
        "user_requirements": "Build a todo app"
    })

    result = result.raw
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    print("\nGemini LLM Result:")
    print(result)

def test_maidsr1_llm(requirement_analyst_maidsr1):
    """Test the MAI-DS-R1 LLM."""
    # Create a task for capturing user intent
    task = Task(
        description="""
        You are a User Intent Analysis AI working for a next-gen software engineering firm.
        Your role is to restructure and enhance the user's initial input into a more comprehensive and clear statement of intent.
        
        Your responsibilities:
        - Analyze the user's initial input
        - Restructure the input into a more comprehensive statement
        - Enhance the description with relevant technical context
        - Maintain the core intent while adding necessary details
        - Format the output for clear understanding

        Output Format:
        ## Original User Input
        - ...

        ## Restructured Intent
        - A clear, comprehensive statement that captures the user's intent with added technical context and necessary details

        ## Key Components
        - Core functionality
        - Technical requirements
        - User experience goals
        - Integration needs
        - Performance expectations

        ## Technical Context
        - Relevant technologies
        - Industry standards
        - Best practices
        - Common patterns
        """,
        agent=requirement_analyst_maidsr1,
        expected_output="A restructured and enhanced version of the user's input that better captures their intent with added technical context and necessary details."
    )
    
    # Create a crew with just the requirement analyst
    crew = Crew(
        agents=[requirement_analyst_maidsr1],
        tasks=[task]
    )
    
    # Test with a simple user input
    result = crew.kickoff(inputs={
        "user_requirements": "Build a todo app"
    })

    result = result.raw

    print("Result: ", result)
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    print("\nMAI-DS-R1 LLM Result:")
    print(result)

def test_llama_llm(requirement_analyst_llama):
    """Test the Llama LLM."""
    # Create a task for capturing user intent
    task = Task(
        description="""
        You are a User Intent Analysis AI working for a next-gen software engineering firm.
        Your role is to restructure and enhance the user's initial input into a more comprehensive and clear statement of intent.
        
        Your responsibilities:
        - Analyze the user's initial input
        - Restructure the input into a more comprehensive statement
        - Enhance the description with relevant technical context
        - Maintain the core intent while adding necessary details
        - Format the output for clear understanding

        Output Format:
        ## Original User Input
        - ...

        ## Restructured Intent
        - A clear, comprehensive statement that captures the user's intent with added technical context and necessary details

        ## Key Components
        - Core functionality
        - Technical requirements
        - User experience goals
        - Integration needs
        - Performance expectations

        ## Technical Context
        - Relevant technologies
        - Industry standards
        - Best practices
        - Common patterns
        """,
        agent=requirement_analyst_llama,
        expected_output="A restructured and enhanced version of the user's input that better captures their intent with added technical context and necessary details."
    )
    
    # Create a crew with just the requirement analyst
    crew = Crew(
        agents=[requirement_analyst_llama],
        tasks=[task]
    )
    
    # Test with a simple user input
    result = crew.kickoff(inputs={
        "user_requirements": "Build a todo app"
    })

    result = result.raw
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    print("\nLlama LLM Result:")
    print(result)

if __name__ == "__main__":
    pytest.main([__file__]) 