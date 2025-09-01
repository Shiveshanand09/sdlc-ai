#pytest sdlc_ai_project/tests/test_github.py -v -s



import os
import pytest
from dotenv import load_dotenv
from crewai_tools import GithubSearchTool
from embedchain import App

# Load environment variables
load_dotenv()

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    if not os.getenv("gh_key"):
        pytest.skip("Github token not set in environment.")

@pytest.fixture(scope="session")
def shared_embedchain_app():
    """Initialize and reuse Embedchain App."""
    app = App(persist_directory="./db/task_mgmt_repo")
    return app

def test_github_code_search():
    tool = GithubSearchTool(
        github_repo='https://github.com/ayushman1024/TASK-Management-System',
        gh_token = os.environ["gh_key"],
        content_types=["code"],
    )
    
    # Execute the search
    result = tool.adapter
    print("Type: ", type(result))
    
    # Basic assertions
    assert result is not None
    # assert isinstance(result, dict)
    # assert len(result) > 0
    
    print("Type: ", type(result))
    # Print the result for inspection
    print("\nScholar Search Result:")
    print(result)



if __name__ == "__main__":
    pytest.main([__file__]) 