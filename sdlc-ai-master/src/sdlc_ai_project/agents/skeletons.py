import os
import json
import inspect
import google.generativeai as genai
from litellm import completion
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
# from crewai_tools import (
#     SerperDevTool,
#     WebScraperTool,
#     CodeAnalysisTool,
#     DocumentationTool,
#     DirectoryReadTool,
#     FileReadTool
# )
load_dotenv()


# -------------------------------
# Agent 3: Code Skeleton Generator
# -------------------------------
@CrewBase
class CodeSkeletonGenerator:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    """
    Code Skeleton Generator:
    This agent automatically creates code templates and boilerplate based on architecture design details.
    It ensures adherence to design specifications and industry best practices.
    """
    
    def __init__(self, architecture_design: str, project_context: str, llm):
        self.architecture_design = architecture_design
        self.project_context = project_context
        self.llm = llm

    @agent
    def skeleton_agent(self) -> Agent:
        return Agent(
            role=f"Code Skeleton Generator for {self.project_context}",
            goal=(
                "Generate robust code templates and boilerplate code based on architecture design and requirements. "
                "Research similar implementations and best practices to inform the code structure. Ensure the code "
                "supports scalability, maintainability, and optimal performance."
            ),
            backstory=(
                "You are an expert in code generation and software engineering, specializing in creating robust, "
                "maintainable code structures. You excel at implementing architectural designs into working code "
                "templates while incorporating industry best practices and modern patterns."
            ),
            description=(
                "Generates code templates and boilerplate code based on architecture design, focusing on "
                "scalability, maintainability, and optimal performance. Researches similar implementations."
            ),
            llm=self.llm,
            # tools=[
            #     SerperDevTool(),  # For searching similar implementations
            #     WebScraperTool(),  # For analyzing similar codebases
            #     CodeAnalysisTool(),  # For analyzing code patterns
            #     DocumentationTool(),  # For analyzing project documentation
            #     DirectoryReadTool(),  # For reading project files
            #     FileReadTool()  # For reading specific files
            # ]
        )

    @task
    def code_research_task(self) -> Task:
        return Task(
            description=(
                f"Research similar implementations and codebases related to {self.project_context}. "
                "Focus on:\n"
                "- Similar projects in the same domain\n"
                "- Common code patterns and structures\n"
                "- Industry best practices and standards\n"
                "- Success stories and lessons learned\n"
                "- Performance optimization techniques\n"
                "Return a JSON object with keys:\n"
                "- 'similar_implementations': List of relevant codebases\n"
                "- 'code_patterns': Common patterns found\n"
                "- 'best_practices': Industry best practices\n"
                "- 'lessons_learned': Key insights from research\n"
                "- 'optimization_techniques': Performance optimizations"
            ),
            expected_output=(
                "A JSON object containing research findings about similar implementations and codebases."
            ),
            agent=self.skeleton_agent()
        )

    @task
    def code_skeleton_task(self) -> Task:
        return Task(
            description=(
                f"Based on the architecture design: {self.architecture_design} and research findings, "
                "generate the main code skeleton. Include:\n"
                "- Project structure and organization\n"
                "- Core module definitions\n"
                "- Interface specifications\n"
                "- Data models and schemas\n"
                "- Error handling patterns\n"
                "- Performance optimization hooks\n"
                "Return a JSON object with keys:\n"
                "- 'project_structure': Directory layout\n"
                "- 'core_modules': Module definitions\n"
                "- 'interfaces': Interface specifications\n"
                "- 'data_models': Data model definitions\n"
                "- 'error_handling': Error handling patterns"
            ),
            expected_output=(
                "A JSON object containing the main code skeleton with project structure and core components."
            ),
            agent=self.skeleton_agent(),
            context=[self.code_research_task()]
        )

    @task
    def module_boilerplate_task(self) -> Task:
        return Task(
            description=(
                "Generate detailed boilerplate code for each module based on the architecture design and code skeleton. "
                "For each module, include:\n"
                "- Class and function definitions\n"
                "- Interface implementations\n"
                "- Data model implementations\n"
                "- Error handling code\n"
                "- Performance optimization code\n"
                "- Integration points\n"
                "Return a JSON object with keys:\n"
                "- 'module_implementations': Module boilerplate code\n"
                "- 'interface_code': Interface implementations\n"
                "- 'data_model_code': Data model implementations\n"
                "- 'error_handling_code': Error handling implementations"
            ),
            expected_output=(
                "A JSON object containing detailed boilerplate code for each module."
            ),
            agent=self.skeleton_agent(),
            context=[self.code_skeleton_task(), self.code_research_task()]
        )

    @task
    def testing_boilerplate_task(self) -> Task:
        return Task(
            description=(
                "Generate testing boilerplate code based on the module implementations and research findings. "
                "Include:\n"
                "- Unit test templates\n"
                "- Integration test templates\n"
                "- Performance test templates\n"
                "- Mock implementations\n"
                "- Test utilities\n"
                "Return a JSON object with keys:\n"
                "- 'unit_tests': Unit test templates\n"
                "- 'integration_tests': Integration test templates\n"
                "- 'performance_tests': Performance test templates\n"
                "- 'test_utilities': Test utility code"
            ),
            expected_output=(
                "A JSON object containing testing boilerplate code and templates."
            ),
            agent=self.skeleton_agent(),
            context=[self.module_boilerplate_task(), self.code_research_task()]
        )

    @task
    def documentation_task(self) -> Task:
        return Task(
            description=(
                "Generate comprehensive documentation based on the code skeleton, module implementations, and research findings. "
                "Include:\n"
                "- API documentation\n"
                "- Architecture overview\n"
                "- Module documentation\n"
                "- Integration guides\n"
                "- Performance guidelines\n"
                "Return a JSON object with keys:\n"
                "- 'api_docs': API documentation\n"
                "- 'architecture_docs': Architecture documentation\n"
                "- 'module_docs': Module documentation\n"
                "- 'integration_docs': Integration guides\n"
                "- 'performance_docs': Performance guidelines"
            ),
            expected_output=(
                "A JSON object containing comprehensive documentation for the codebase."
            ),
            agent=self.skeleton_agent(),
            context=[self.module_boilerplate_task(), self.testing_boilerplate_task(), self.code_research_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.skeleton_agent()
            ],
            tasks=[
                self.code_research_task(),
                self.code_skeleton_task(),
                self.module_boilerplate_task(),
                self.testing_boilerplate_task(),
                self.documentation_task()
            ],
            process=Process.sequential
        )
