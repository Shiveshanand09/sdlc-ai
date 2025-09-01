from crewai_tools import GithubSearchTool
import os

tool = GithubSearchTool(
        github_repo='https://github.com/ayushman1024/TASK-Management-System',
        gh_token = os.environ["gh_key"],
        content_types=["code"]  # Specify the required content types
    )