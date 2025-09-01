import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
import google.generativeai as genai
from sdlc_ai_project.tools import code_research_tool, document_research_tool

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@CrewBase
class KnowledgeBaseAgent:
    """
    Knowledge Base Agent:
    This CrewBase class defines an autonomous agent that:
      1. Researches similar systems online.
      2. Scrapes official documentation and tutorials.
      3. Extracts code patterns from open-source repositories.
      4. Reads local project files for context.
      5. Aggregates all findings into a structured JSON knowledge base.
    """

    def __init__(self, user_requirements: str, project_context: str, llm):
        self.user_requirements = user_requirements
        self.project_context = project_context
        self.llm = llm

    @agent
    def kb_research_agent(self) -> Agent:
        return Agent(
            role=f"Knowledge Research Specialist for {self.project_context}",
            goal=(
                "Automate the gathering of domain-specific knowledge "
                "and code examples to build a comprehensive knowledge base."
            ),
            backstory=(
                "You are an AI researcher with expertise in information retrieval, "
                "documentation analysis, and code pattern recognition."
            ),
            description=(
                "Conducts a blend of web search, scraping, and code analysis "
                "to prepare reference material for software development."
            ),
            llm=self.llm,
            # tools=[code_research_tool, document_research_tool]
        )

    @task
    def research_similar_projects(self) -> Task:
        return Task(
            description=(
                f"Using the user requirements: {self.user_requirements}, "
                f"research similar open-source projects and software in the domain of {self.project_context}. "
                "Return JSON with keys: 'projects': [{ 'name', 'repo_url', 'summary' }]."
                "Validate the urls are correct"
            ),
            expected_output="JSON list of similar projects with metadata.",
            agent=self.kb_research_agent()
        )

    @task
    def gather_documentation(self) -> Task:
        return Task(
            description=(
                "Scrape official docs, tutorials, and API references for the identified projects and technologies including language documentation and frameworks documentation. "
                "Return JSON with 'docs': [{ 'title', 'url', 'snippet' }]."
            ),
            expected_output="JSON list of documentation snippets.",
            agent=self.kb_research_agent(),
            context=[self.research_similar_projects()]
        )

    @task
    def collect_code_samples(self) -> Task:
        return Task(
            description=(
                "Analyze the code repositories and extract representative code samples or patterns. "
                "Return JSON with 'code_samples': [{ 'file_path', 'pattern', 'example' }]."
            ),
            expected_output="JSON list of code patterns and examples.",
            agent=self.kb_research_agent(),
            context=[self.research_similar_projects()],
        )

    @task
    def build_knowledge_base(self) -> Task:
        return Task(
            description=(
                "Integrate the outputs from research_similar_projects, gather_documentation, "
                "and collect_code_samples into a single JSON knowledge_base object with keys: "
                "'projects', 'documentation', 'code_samples'."
            ),
            expected_output="A consolidated JSON knowledge_base.",
            agent=self.kb_research_agent(),
            context=[
                self.research_similar_projects(),
                self.gather_documentation(),
                self.collect_code_samples()
            ]
        )
    
    @task
    def finalize_tech_stack(self) -> Task:
        return Task(
            description=(
                "Analyze the gathered knowledge base, including similar projects, documentation, "
                "and code samples, to propose a finalized tech stack for the project. "
                "Return JSON with 'tech_stack': [{ 'technology', 'reason' }]."
            ),
            expected_output="JSON list of technologies with reasons for selection.",
            agent=self.kb_research_agent(),
            context=[
                self.build_knowledge_base()
            ]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.kb_research_agent()],
            tasks=[
                self.research_similar_projects(),
                self.gather_documentation(),
                self.collect_code_samples(),
                self.build_knowledge_base(),
                self.finalize_tech_stack()
            ],
            # memory=True,
            process=Process.sequential,
            # long_term_memory=LongTermMemory(
            #     storage = LTMSQLiteStorage(
            #         db_path='./db/default_ragtool_db'
            #     )
            # ),
            # short_term_memory=ShortTermMemory(
            #     storage = LTMSQLiteStorage(
            #         db_path='./db/default_ragtool_db'
            #     )
            # ),
            # entity_memory=EntityMemory(
            #     storage = LTMSQLiteStorage(
            #         db_path='./db/default_ragtool_db'
            #     )
            # )
        )
