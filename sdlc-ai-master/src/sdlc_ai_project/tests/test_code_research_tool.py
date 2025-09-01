import os
import pytest
from unittest.mock import patch, MagicMock
from sdlc_ai_project.tools import code_research_tool

@pytest.fixture(autouse=True)
def setup_environment():
    """Setup test environment variables."""
    os.environ["gh_key"] = "mock_github_token"

@patch("sdlc_ai_project.tools.SerperDevTool")
@patch("sdlc_ai_project.tools.GithubSearchTool")
@patch("sdlc_ai_project.tools.App")
def test_code_research_tool(mock_app, mock_github_search_tool, mock_serper_dev_tool):
    # Mock SerperDevTool
    mock_serper_dev_tool.return_value = {
        "organic": [
            {
                "title": "Top 8 Open-Source Projects",
                "link": "https://github.com/kanboard/kanboard",
                "snippet": "Kanboard is a lightweight Kanban tool.",
                "position": 1,
                "sitelinks": []
            },
            {
                "title": "HELP! I need a task tool",
                "link": "https://github.com/JordanKnott/taskcafe",
                "snippet": "Taskcafe is another task management app.",
                "position": 2,
                "sitelinks": []
            }
        ]
    }

    # Mock GithubSearchTool and App
    mock_github_search_tool.return_value = MagicMock()
    mock_app_instance = MagicMock()
    mock_app_instance.query.return_value = "Mocked relevant context"
    mock_app.from_config.return_value = mock_app_instance

    # Call the function
    query = "A simple web application for task management"
    result = code_research_tool(query)

    # Assertions
    assert result == "Mocked relevant context"

    # Verify
    mock_serper_dev_tool.assert_called_once_with(
        search_url=("Similar projects for ", query),
        n_results=2
    )
    assert mock_github_search_tool.call_count == 2
    mock_app.from_config.assert_called_once()
    mock_app_instance.query.assert_called_once_with(query)

    # Print result
    print("\nTest Result:")
    print(result)
