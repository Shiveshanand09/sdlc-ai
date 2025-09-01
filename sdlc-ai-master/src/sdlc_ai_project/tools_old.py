from crewai.tools import BaseTool
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool,
    CodeDocsSearchTool,
    CSVSearchTool,
    DOCXSearchTool,
    JSONSearchTool,
    PDFSearchTool,
    TXTSearchTool,
    XMLSearchTool,
    GithubSearchTool,
    ScrapeWebsiteTool
)
from typing import Optional, Dict, List, Any
import os
import json
import requests
from pathlib import Path
import time
from datetime import datetime, timedelta
import hashlib
import pickle
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import Field, SkipValidation
from crewai import LLM

# Load environment variables
load_dotenv()

# Environment variables with defaults
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # Empty string as default
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "60"))
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
MAX_AGENTS = int(os.getenv("MAX_AGENTS", "10"))
MAX_TASKS_PER_AGENT = int(os.getenv("MAX_TASKS_PER_AGENT", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

def get_configured_llm() -> LLM:
    """Get a configured LLM instance using Gemini."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    return LLM(
        model="gemini/gemini-1.5-flash",
        api_key=gemini_api_key
    )

# Initialize the LLM
llm = get_configured_llm()

# Cache management
class CacheManager:
    def __init__(self, cache_dir: str = CACHE_DIR, ttl: int = CACHE_TTL):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl  # Time to live in seconds
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Any]:
        cache_key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            # Check if cache is still valid
            if time.time() - cache_file.stat().st_mtime < self.ttl:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def set(self, query: str, data: Any):
        cache_key = self.get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

# Rate limiting and token management
class RateLimiter:
    def __init__(self, requests_per_minute: int = RATE_LIMIT):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    def can_make_request(self) -> bool:
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < timedelta(minutes=1)]
        return len(self.requests) < self.requests_per_minute
    
    def add_request(self):
        self.requests.append(datetime.now())

class TokenManager:
    def __init__(self, max_tokens: int = MAX_TOKENS):
        self.max_tokens = max_tokens
        self.current_tokens = 0
    
    def can_add_tokens(self, tokens: int) -> bool:
        return self.current_tokens + tokens <= self.max_tokens
    
    def add_tokens(self, tokens: int):
        self.current_tokens += tokens
    
    def reset(self):
        self.current_tokens = 0

# Agent limit management
class AgentLimitManager:
    def __init__(self, max_agents: int = MAX_AGENTS, max_tasks_per_agent: int = MAX_TASKS_PER_AGENT):
        self.max_agents = max_agents
        self.max_tasks_per_agent = max_tasks_per_agent
        self.agent_tasks = {}
    
    def can_add_agent(self) -> bool:
        return len(self.agent_tasks) < self.max_agents
    
    def can_add_task(self, agent_id: str) -> bool:
        return self.agent_tasks.get(agent_id, 0) < self.max_tasks_per_agent
    
    def add_agent(self, agent_id: str):
        if self.can_add_agent():
            self.agent_tasks[agent_id] = 0
    
    def add_task(self, agent_id: str):
        if agent_id in self.agent_tasks:
            self.agent_tasks[agent_id] += 1
    
    def remove_agent(self, agent_id: str):
        if agent_id in self.agent_tasks:
            del self.agent_tasks[agent_id]

# Initialize managers
rate_limiter = RateLimiter(requests_per_minute=RATE_LIMIT)
token_manager = TokenManager(max_tokens=MAX_TOKENS)
cache_manager = CacheManager(cache_dir=CACHE_DIR, ttl=CACHE_TTL)
agent_limit_manager = AgentLimitManager(max_agents=MAX_AGENTS, max_tasks_per_agent=MAX_TASKS_PER_AGENT)

# Documentation Tools
class DocumentationTool(BaseTool):
    name: str = "documentation_tool"
    description: str = "Tool for accessing and following documentation standards"
    serper_tool: SkipValidation[SerperDevTool] = Field(default_factory=SerperDevTool)
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(
        default_factory=lambda: CodeDocsSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
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
    )
    github_tool: SkipValidation[GithubSearchTool] = Field(
        default_factory=lambda: GithubSearchTool(
            gh_token=GITHUB_TOKEN or "dummy_token",
            content_types=['code', 'issue'],
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
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
    )
    
    def _run(self, query: str) -> str:
        # First, use Serper to find relevant documentation URLs
        search_result = self.serper_tool.run(search_query=f"{query} documentation")
        if not search_result or "organic" not in search_result:
            return "No documentation found"
        
        # Get the first link from search results
        if not search_result["organic"] or "link" not in search_result["organic"][0]:
            return "No documentation URLs found"
        
        doc_url = search_result["organic"][0]["link"]
        
        # Use CodeDocsSearchTool to search in the found documentation
        try:
            self.code_docs_tool.docs_url = doc_url
            result = self.code_docs_tool.run(query)
            if result:
                code_docs_result = f"From {doc_url}:\n{result}"
            else:
                code_docs_result = f"No results found in {doc_url}"
        except Exception as e:
            code_docs_result = f"Error searching {doc_url}: {str(e)}"
        
        return f"Documentation search results:\n{code_docs_result}"

class ArchitectureDocumentationTool(BaseTool):
    name: str = "architecture_documentation_tool"
    description: str = "Tool for accessing architectural documentation standards"
    serper_tool: SkipValidation[SerperDevTool] = Field(default_factory=SerperDevTool)
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(
        default_factory=lambda: CodeDocsSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
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
    )
    web_tool: SkipValidation[WebsiteSearchTool] = Field(
        default_factory=lambda: WebsiteSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
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
    )
    
    def _run(self, query: str) -> str:
        # First, use Serper to find relevant architecture documentation URLs
        search_result = self.serper_tool.run(search_query=f"{query} architecture documentation")
        if not search_result or "organic" not in search_result:
            return "No architecture documentation found"
        
        # Get the first link from search results
        if not search_result["organic"] or "link" not in search_result["organic"][0]:
            return "No architecture documentation URLs found"
        
        doc_url = search_result["organic"][0]["link"]
        
        # Use CodeDocsSearchTool to search in the found documentation
        try:
            self.code_docs_tool.docs_url = doc_url
            result = self.code_docs_tool._run(query)
            if result:
                code_docs_result = f"From {doc_url}:\n{result}"
            else:
                code_docs_result = f"No results found in {doc_url}"
        except Exception as e:
            code_docs_result = f"Error searching {doc_url}: {str(e)}"
        
        # Also search web for architectural patterns
        web_result = self.web_tool._run(query)
        
        return f"Architectural documentation from code docs:\n{code_docs_result}\n\nWeb resources: {web_result}"

class TechStackDocumentationTool(BaseTool):
    name: str = "tech_stack_documentation_tool"
    description: str = "Tool for accessing tech stack documentation and best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(
        default_factory=lambda: CodeDocsSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
            )
        )
    )
    web_tool: SkipValidation[WebsiteSearchTool] = Field(
        default_factory=lambda: WebsiteSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
            )
        )
    )
    github_tool: SkipValidation[GithubSearchTool] = Field(
        default_factory=lambda: GithubSearchTool(
            gh_token=GITHUB_TOKEN or "dummy_token",
            content_types=['code', 'issue'],
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-pro",
                        temperature=0.7,
                        max_tokens=MAX_TOKENS,
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        task_type="retrieval_document",
                        api_key=GOOGLE_API_KEY,
                    ),
                ),
            )
        )
    )
    search_tool: SkipValidation[BraveSearchTool] = Field(default_factory=BraveSearchTool)
    
    @lru_cache(maxsize=100)
    def _cached_search(self, query: str, tool: Any) -> str:
        return tool._run(query)
    
    def _run(self, query: str) -> str:
        # Check cache first
        cached_result = cache_manager.get(query)
        if cached_result:
            return cached_result
        
        # Extract tech stack components from the query
        tech_stack = self._extract_tech_stack(query)
        
        results = []
        for tech in tech_stack:
            # Use cached searches where possible
            docs_result = self._cached_search(f"{tech} documentation best practices", self.code_docs_tool)
            github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self._cached_search(f"{tech} example implementation", self.github_tool)
            web_result = self._cached_search(f"{tech} best practices", self.web_tool)
            search_result = self._cached_search(f"{tech} common issues and solutions", self.search_tool)
            
            results.append(f"""
            Tech Stack Component: {tech}
            Documentation: {docs_result}
            GitHub Examples: {github_result}
            Best Practices: {web_result}
            Common Issues: {search_result}
            """)
        
        result = "\n".join(results)
        # Cache the result
        cache_manager.set(query, result)
        return result
    
    def _extract_tech_stack(self, query: str) -> List[str]:
        # This is a simple implementation. You might want to use NLP or other methods
        # to extract tech stack components more accurately
        common_techs = [
            "python", "javascript", "typescript", "java", "c#", "go", "rust",
            "react", "angular", "vue", "django", "flask", "spring", "express",
            "postgresql", "mysql", "mongodb", "redis", "docker", "kubernetes"
        ]
        return [tech for tech in common_techs if tech.lower() in query.lower()]

class LanguageDocumentationTool(BaseTool):
    name: str = "language_documentation_tool"
    description: str = "Tool for accessing programming language documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for language-specific information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Language documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

class FrameworkDocumentationTool(BaseTool):
    name: str = "framework_documentation_tool"
    description: str = "Tool for accessing framework documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for framework information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Framework documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

class TestingFrameworkDocumentationTool(BaseTool):
    name: str = "testing_framework_documentation_tool"
    description: str = "Tool for accessing testing framework documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for testing framework information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Testing framework documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

class DeploymentDocumentationTool(BaseTool):
    name: str = "deployment_documentation_tool"
    description: str = "Tool for accessing deployment documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for deployment information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Deployment documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

class MonitoringDocumentationTool(BaseTool):
    name: str = "monitoring_documentation_tool"
    description: str = "Tool for accessing monitoring documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for monitoring information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Monitoring documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

class MaintenanceDocumentationTool(BaseTool):
    name: str = "maintenance_documentation_tool"
    description: str = "Tool for accessing maintenance documentation"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for maintenance information
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Maintenance documentation from code docs: {code_docs_result}\nWeb resources: {web_result}"

# Best Practices Tools
class BestPracticesTool(BaseTool):
    name: str = "best_practices_tool"
    description: str = "Tool for accessing industry best practices"
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    
    def _run(self, query: str) -> str:
        # Search web and GitHub for best practices
        web_result = self.web_tool._run(query)
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        return f"Best practices from web: {web_result}\nGitHub resources: {github_result}"

class DesignPatternsTool(BaseTool):
    name: str = "design_patterns_tool"
    description: str = "Tool for accessing design patterns"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for design patterns
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Design patterns from code docs: {code_docs_result}\nWeb resources: {web_result}"

class CodingStandardsTool(BaseTool):
    name: str = "coding_standards_tool"
    description: str = "Tool for accessing coding standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    
    def _run(self, query: str) -> str:
        # Search code documentation and GitHub for coding standards
        code_docs_result = self.code_docs_tool._run(query)
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        return f"Coding standards from code docs: {code_docs_result}\nGitHub resources: {github_result}"

class TestingBestPracticesTool(BaseTool):
    name: str = "testing_best_practices_tool"
    description: str = "Tool for accessing testing best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for testing best practices
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Testing best practices from code docs: {code_docs_result}\nWeb resources: {web_result}"

class DebuggingBestPracticesTool(BaseTool):
    name: str = "debugging_best_practices_tool"
    description: str = "Tool for accessing debugging best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for debugging best practices
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Debugging best practices from code docs: {code_docs_result}\nWeb resources: {web_result}"

class DevOpsBestPracticesTool(BaseTool):
    name: str = "devops_best_practices_tool"
    description: str = "Tool for accessing DevOps best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for DevOps best practices
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"DevOps best practices from code docs: {code_docs_result}\nWeb resources: {web_result}"

class ObservabilityBestPracticesTool(BaseTool):
    name: str = "observability_best_practices_tool"
    description: str = "Tool for accessing observability best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for observability best practices
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Observability best practices from code docs: {code_docs_result}\nWeb resources: {web_result}"

class OptimizationBestPracticesTool(BaseTool):
    name: str = "optimization_best_practices_tool"
    description: str = "Tool for accessing optimization best practices"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for optimization best practices
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Optimization best practices from code docs: {code_docs_result}\nWeb resources: {web_result}"

# Standards Tools
class ProjectStructureTool(BaseTool):
    name: str = "project_structure_tool"
    description: str = "Tool for accessing project structure standards"
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search GitHub and web for project structure standards
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Project structure standards from GitHub: {github_result}\nWeb resources: {web_result}"

class CoverageStandardsTool(BaseTool):
    name: str = "coverage_standards_tool"
    description: str = "Tool for accessing coverage standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for coverage standards
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Coverage standards from code docs: {code_docs_result}\nWeb resources: {web_result}"

class CodeQualityStandardsTool(BaseTool):
    name: str = "code_quality_standards_tool"
    description: str = "Tool for accessing code quality standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    
    def _run(self, query: str) -> str:
        # Search code documentation and GitHub for code quality standards
        code_docs_result = self.code_docs_tool._run(query)
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        return f"Code quality standards from code docs: {code_docs_result}\nGitHub resources: {github_result}"

class SecurityStandardsTool(BaseTool):
    name: str = "security_standards_tool"
    description: str = "Tool for accessing security standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for security standards
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Security standards from code docs: {code_docs_result}\nWeb resources: {web_result}"

class AlertingStandardsTool(BaseTool):
    name: str = "alerting_standards_tool"
    description: str = "Tool for accessing alerting standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    
    def _run(self, query: str) -> str:
        # Search code documentation and web for alerting standards
        code_docs_result = self.code_docs_tool._run(query)
        web_result = self.web_tool._run(query)
        return f"Alerting standards from code docs: {code_docs_result}\nWeb resources: {web_result}"

# Specialized Tools
class RefactoringGuidelinesTool(BaseTool):
    name: str = "refactoring_guidelines_tool"
    description: str = "Tool for accessing refactoring guidelines"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    
    def _run(self, query: str) -> str:
        # Search code documentation and GitHub for refactoring guidelines
        code_docs_result = self.code_docs_tool._run(query)
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        return f"Refactoring guidelines from code docs: {code_docs_result}\nGitHub resources: {github_result}"

class DependencyManagementTool(BaseTool):
    name: str = "dependency_management_tool"
    description: str = "Tool for accessing dependency management standards"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    
    def _run(self, query: str) -> str:
        # Search code documentation and GitHub for dependency management standards
        code_docs_result = self.code_docs_tool._run(query)
        github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self.github_tool._run(query)
        return f"Dependency management standards from code docs: {code_docs_result}\nGitHub resources: {github_result}"

class BugAnalysisTool(BaseTool):
    name: str = "bug_analysis_tool"
    description: str = "Tool for analyzing bugs and finding solutions from documentation and community resources"
    code_docs_tool: SkipValidation[CodeDocsSearchTool] = Field(default_factory=lambda: CodeDocsSearchTool(llm=llm))
    web_tool: SkipValidation[WebsiteSearchTool] = Field(default_factory=lambda: WebsiteSearchTool(llm=llm))
    github_tool: SkipValidation[GithubSearchTool] = Field(default_factory=lambda: GithubSearchTool(
        gh_token=GITHUB_TOKEN or "dummy_token",  # Use dummy token if none provided
        content_types=['code', 'issue'],
        llm=llm
    ))
    search_tool: SkipValidation[BraveSearchTool] = Field(default_factory=BraveSearchTool)
    
    @lru_cache(maxsize=100)
    def _cached_search(self, query: str, tool: Any) -> str:
        return tool._run(query)
    
    def _run(self, query: str) -> str:
        # Check cache first
        cached_result = cache_manager.get(query)
        if cached_result:
            return cached_result
        
        if not rate_limiter.can_make_request():
            time.sleep(1)  # Wait if rate limit is reached
        
        # Extract bug details from the query
        bug_details = self._extract_bug_details(query)
        
        results = []
        for bug in bug_details:
            # Check token limit before making requests
            if not token_manager.can_add_tokens(1000):  # Estimate 1000 tokens per bug
                break
            
            # Use cached searches where possible
            docs_result = self._cached_search(f"{bug['type']} {bug['description']} solution", self.code_docs_tool)
            github_result = "GitHub search disabled (no token provided)" if not GITHUB_TOKEN else self._cached_search(f"{bug['type']} {bug['description']} issue", self.github_tool)
            web_result = self._cached_search(f"{bug['type']} {bug['description']} fix", self.web_tool)
            search_result = self._cached_search(f"{bug['type']} {bug['description']} error", self.search_tool)
            
            results.append(f"""
            Bug Type: {bug['type']}
            Description: {bug['description']}
            Documentation Solutions: {docs_result}
            GitHub Issues: {github_result}
            Web Solutions: {web_result}
            Error Analysis: {search_result}
            """)
            
            # Add tokens used
            token_manager.add_tokens(1000)
            rate_limiter.add_request()
        
        result = "\n".join(results)
        # Cache the result
        cache_manager.set(query, result)
        return result
    
    def _extract_bug_details(self, query: str) -> List[Dict[str, str]]:
        # This is a simple implementation. You might want to use NLP or other methods
        # to extract bug details more accurately
        return [{
            'type': 'error',
            'description': query
        }]

# Tool Factory
def get_tool(tool_name: str) -> Optional[BaseTool]:
    """Factory function to get a tool instance by name"""
    tool_classes = {
        # Documentation Tools
        "documentation_tool": DocumentationTool,
        "architecture_documentation_tool": ArchitectureDocumentationTool,
        "tech_stack_documentation_tool": TechStackDocumentationTool,
        "language_documentation_tool": LanguageDocumentationTool,
        "framework_documentation_tool": FrameworkDocumentationTool,
        "testing_framework_documentation_tool": TestingFrameworkDocumentationTool,
        "deployment_documentation_tool": DeploymentDocumentationTool,
        "monitoring_documentation_tool": MonitoringDocumentationTool,
        "maintenance_documentation_tool": MaintenanceDocumentationTool,
        
        # Best Practices Tools
        "best_practices_tool": BestPracticesTool,
        "design_patterns_tool": DesignPatternsTool,
        "coding_standards_tool": CodingStandardsTool,
        "testing_best_practices_tool": TestingBestPracticesTool,
        "debugging_best_practices_tool": DebuggingBestPracticesTool,
        "devops_best_practices_tool": DevOpsBestPracticesTool,
        "observability_best_practices_tool": ObservabilityBestPracticesTool,
        "optimization_best_practices_tool": OptimizationBestPracticesTool,
        
        # Standards Tools
        "project_structure_tool": ProjectStructureTool,
        "coverage_standards_tool": CoverageStandardsTool,
        "code_quality_standards_tool": CodeQualityStandardsTool,
        "security_standards_tool": SecurityStandardsTool,
        "alerting_standards_tool": AlertingStandardsTool,
        
        # Specialized Tools
        "refactoring_guidelines_tool": RefactoringGuidelinesTool,
        "dependency_management_tool": DependencyManagementTool,
        "bug_analysis_tool": BugAnalysisTool,
    }
    
    tool_class = tool_classes.get(tool_name)
    if tool_class:
        return tool_class()
    return None