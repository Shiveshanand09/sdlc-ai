import os
import pytest
from dotenv import load_dotenv
from sdlc_ai_project.tools_old import get_tool, get_configured_llm
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    # Check for required API keys
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")
    if not os.getenv("gh_key"):
        pytest.skip("GITHUB_TOKEN not set in environment")
    if not os.getenv("SERPER_API_KEY"):
        pytest.skip("SERPER_API_KEY not set in environment. Please register at serper.dev and set the API key.")

def test_llm_configuration():
    """Test that the LLM is properly configured with Gemini."""
    llm = get_configured_llm()
    assert llm is not None
    assert llm.model == "gemini/gemini-1.5-flash"
    assert llm.api_key == os.getenv("GEMINI_API_KEY")

def test_github_search_tool():
    """Test GitHub search tool functionality."""
    tool = get_tool("documentation_tool")
    assert tool is not None
    
    # Test GitHub search with a specific repository
    result = tool.github_tool._run("python best practices")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with content type filtering
    result = tool.github_tool._run("python best practices", content_types=['code'])
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_serper_search_tool():
    """Test Serper search tool functionality."""
    tool = get_tool("tech_stack_documentation_tool")
    assert tool is not None
    
    # Test basic search
    result = tool.search_tool.run(search_query="python best practices")
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Verify the result format
    assert "searchParameters" in result
    assert "organic" in result
    assert "relatedSearches" in result
    assert "credits" in result
    
    # Test with custom parameters
    result = tool.search_tool.run(
        search_query="python best practices",
        country="US",
        n_results=2
    )
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Verify search parameters
    assert result["searchParameters"]["q"] == "python best practices"
    assert result["searchParameters"]["num"] == 2
    assert result["searchParameters"]["engine"] == "google"
    
    # Verify organic results
    assert len(result["organic"]) > 0
    first_result = result["organic"][0]
    assert "title" in first_result
    assert "link" in first_result
    assert "snippet" in first_result

def test_code_docs_search_tool():
    """Test code documentation search tool functionality."""
    tool = get_tool("documentation_tool")
    assert tool is not None
    
    # Test basic search
    result = tool.code_docs_tool._run("python documentation best practices")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with custom configuration
    result = tool.code_docs_tool._run(
        "python documentation best practices",
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini-1.5-pro",
                    temperature=0.7,
                    api_key=os.getenv("GEMINI_API_KEY"),
                ),
            ),
        )
    )
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_website_search_tool():
    """Test website search tool functionality."""
    tool = get_tool("architecture_documentation_tool")
    assert tool is not None
    
    # Test basic search
    result = tool.web_tool._run("python architecture best practices")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with custom configuration
    result = tool.web_tool._run(
        "python architecture best practices",
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini-1.5-pro",
                    temperature=0.7,
                    api_key=os.getenv("GEMINI_API_KEY"),
                ),
            ),
        )
    )
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

def test_tool_error_handling():
    """Test error handling in tools."""
    # Test with invalid tool name
    invalid_tool = get_tool("invalid_tool_name")
    assert invalid_tool is None
    
    # Test with empty query
    tool = get_tool("documentation_tool")
    result = tool._run("")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Test with invalid GitHub token
    tool.github_tool.gh_token = "invalid_token"
    result = tool.github_tool._run("python best practices")
    assert "GitHub search disabled" in result
    
    # Test with invalid Serper API key
    tool = get_tool("tech_stack_documentation_tool")
    tool.search_tool.api_key = "invalid_key"
    with pytest.raises(Exception):
        tool.search_tool.run(search_query="Python")

def test_tool_caching():
    """Test caching functionality."""
    tool = get_tool("documentation_tool")
    
    # First request
    result1 = tool._run("What are Python best practices?")
    assert result1 is not None
    
    # Second request with same query (should use cache)
    result2 = tool._run("What are Python best practices?")
    assert result2 is not None
    
    # Results should be identical if caching is working
    assert result1 == result2

def test_tool_configuration():
    """Test tool configuration options."""
    tool = get_tool("tech_stack_documentation_tool")
    assert tool is not None
    
    # Test with different model configurations
    result = tool.search_tool.run(
        search_query="python best practices",
        country="US",
        n_results=2
    )
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Verify the result format
    assert "searchParameters" in result
    assert "organic" in result
    assert "relatedSearches" in result
    assert "credits" in result

if __name__ == "__main__":
    pytest.main([__file__]) 