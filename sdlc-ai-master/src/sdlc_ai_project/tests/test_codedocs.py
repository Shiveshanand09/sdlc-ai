import os
import pytest
from dotenv import load_dotenv
from crewai_tools import CodeDocsSearchTool

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set in environment")

# def test_codedocs_general_search():
#     """Test CodeDocsSearchTool for general documentation search."""
#     # Initialize the tool without specific docs_url
#     tool = CodeDocsSearchTool()
    
#     # Test search in any code documentation
#     result = tool.run("How to use search tool")
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     print("\nGeneral Search Result:")
#     print(result)

# def test_codedocs_specific_site():
#     """Test CodeDocsSearchTool with specific documentation site."""
#     # Initialize the tool with specific docs_url
#     tool = CodeDocsSearchTool(docs_url='https://docs.crewai.com/reference')
    
#     # Test search in specific documentation site
#     result = tool.run("How to use search tool")
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     print("\nSpecific Site Search Result:")
#     print(result)

def test_codedocs_custom_config():
    """Test CodeDocsSearchTool with custom model configuration."""
    # Initialize the tool with custom configuration
    tool = CodeDocsSearchTool(
        config=dict(
            llm=dict(
                provider="google",
                config=dict(
                    model="gemini-1.5-pro",
                    temperature=0.7,
                    api_key=os.getenv("GEMINI_API_KEY"),
                ),
            ),
            embedder=dict(
                provider="google",
                config=dict(
                    model="gemini-embedding-exp-03-07",
                    task_type="retrieval_document",
                ),
            ),
        )
    )
    
    # Test search with custom configuration
    result = tool.run("https://nextjs.org/docs")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    
    print("\nCustom Config Search Result:")
    print(result)

# def test_codedocs_error_handling():
#     """Test CodeDocsSearchTool error handling."""
#     # Test with empty query
#     tool = CodeDocsSearchTool()
#     result = tool.run("")
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     print("\nEmpty Query Result:")
#     print(result)
    
#     # Test with invalid docs_url
#     tool = CodeDocsSearchTool(docs_url='https://invalid-docs-url.com')
#     result = tool.run("How to use search tool")
#     assert result is not None
#     assert isinstance(result, str)
#     assert len(result) > 0
    
#     print("\nInvalid Docs URL Result:")
#     print(result)

if __name__ == "__main__":
    pytest.main([__file__]) 