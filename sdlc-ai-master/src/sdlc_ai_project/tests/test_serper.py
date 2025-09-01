#pytest sdlc_ai_project/tests/test_requirement_analyst.py -v -s



import os
import pytest
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("SERPER_API_KEY"):
        pytest.skip("SERPER_API_KEY not set in environment. Please register at serper.dev and set the API key.")

def test_serper_scholar_search():
    """Test Serper scholar search functionality."""
    # Initialize the tool with scholar search URL and result count
    tool = SerperDevTool(
        search_url="Github Projects for A simple web application for task management",
        n_results=2
    )
    
    # Execute the search
    result = tool.run(search_query="Github Projects for A simple web application for task management")
    
    # Basic assertions
    assert result is not None
    assert isinstance(result, dict)
    assert len(result) > 0
    
    print("Type: ", type(result))
    
    # Verify the result format
    # assert "searchParameters" in result
    # assert "organic" in result
    # assert "relatedSearches" in result
    # assert "credits" in result
    
    # # Verify search parameters
    # assert result["searchParameters"]["q"] == "ChatGPT"
    # assert result["searchParameters"]["num"] == 2
    # assert result["searchParameters"]["engine"] == "google"
    
    # Verify organic results
    # assert len(result["organic"]) > 0
    # first_result = result["organic"][0]
    # assert "title" in first_result
    # assert "link" in first_result
    # assert "snippet" in first_result
    
    # Print the result for inspection
    print("\nScholar Search Result:")
    print(result['organic'][0]['link'])

# def test_serper_basic_search():
#     """Test basic Serper search functionality."""
#     # Initialize the tool with default settings
#     tool = SerperDevTool()
    
#     # Execute the search
#     result = tool.run(search_query="Python best practices")
    
#     # Basic assertions
#     assert result is not None
#     assert isinstance(result, dict)
#     assert len(result) > 0
    
#     # Verify the result format
#     assert "searchParameters" in result
#     assert "organic" in result
#     assert "relatedSearches" in result
#     assert "credits" in result
    
#     # Print the result for inspection
#     print("\nBasic Search Result:")
#     print(result)

# def test_serper_custom_parameters():
#     """Test Serper with custom parameters."""
#     # Initialize the tool with custom parameters
#     tool = SerperDevTool(
#         search_url="https://google.serper.dev/search",
#         n_results=3
#     )
    
#     # Execute the search with additional parameters
#     result = tool.run(
#         search_query="Python programming",
#         country="US",
#         location="New York, NY, USA"
#     )
    
#     # Basic assertions
#     assert result is not None
#     assert isinstance(result, dict)
#     assert len(result) > 0
    
#     # Verify the result format
#     assert "searchParameters" in result
#     assert "organic" in result
#     assert "relatedSearches" in result
#     assert "credits" in result
    
#     # Print the result for inspection
#     print("\nCustom Parameters Search Result:")
#     print(result)

if __name__ == "__main__":
    pytest.main([__file__]) 