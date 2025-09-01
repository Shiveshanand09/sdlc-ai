import os
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
# from crewai_tools import (
#     CodeAnalysisTool,
#     DocumentationTool,
#     TestingFrameworkTool,
#     DebuggingTool
# )
from sdlc_ai_project.agents.skeletons import CodeSkeletonGenerator

load_dotenv()

@CrewBase
class CodeGenerator:
    """
    Code Agent:
    This agent takes the output of the CodeSkeletonGenerator and generates production-ready code
    for each module, ensuring alignment with the architecture design and best practices.
    """

    def __init__(self, architecture_design: str, project_context: str, skeletons, module_boilerplate, llm):
        self.architecture_design = architecture_design
        self.project_context = project_context
        self.skeletons = skeletons
        self.module_boilerplate = module_boilerplate
        self.llm = llm

    @agent
    def code_agent(self) -> Agent:
        return Agent(
            role=f"Code Generation Specialist for {self.project_context}",
            goal=(
                "Generate production-ready code for each module based on the code skeleton and architecture design. "
                "Ensure the code adheres to clean code principles, design patterns, and industry best practices."
            ),
            backstory=(
                "You are a seasoned software engineer with expertise in writing clean, maintainable, and scalable code. "
                "You excel at transforming skeletons into fully functional systems while ensuring alignment with "
                "architectural designs and best practices."
            ),
            description=(
                "Generates production-ready code for each module, focusing on maintainability, scalability, and performance."
            ),
            llm=self.llm,
            # tools=[
            #     CodeAnalysisTool(),  # For analyzing and improving code quality
            #     DocumentationTool(),  # For generating inline documentation
            #     TestingFrameworkTool(),  # For ensuring testability
            #     DebuggingTool()  # For identifying and resolving potential issues
            # ]
        )

    @task
    def generate_code_task(self) -> Task:
        return Task(
            description=(
                f"Using the code skeleton: {self.skeletons} {self.module_boilerplate} and architecture design: {self.architecture_design}, generate production-ready code for each module. "
                "Include:\n"
                "- Implementation of core logic\n"
                "- Error handling and logging\n"
                "- Integration with other modules\n"
                "- Adherence to design patterns\n"
                "- Inline documentation\n"
                "Return a JSON object with keys:\n"
                "- 'module_code': Code for each module\n"
                "- 'integration_code': Code for module integration\n"
                "- 'error_handling': Error handling implementations\n"
                "- 'logging': Logging implementations"
            ),
            expected_output=(
                "A JSON object containing production-ready code for each module and integration points."
            ),
            agent=self.code_agent(),
        )

    @task
    def validate_code_task(self) -> Task:
        return Task(
            description=(
                "Validate the generated code against the architecture design and best practices. Ensure:\n"
                "- Alignment with the architecture design\n"
                "- Adherence to clean code principles\n"
                "- Proper error handling and logging\n"
                "- Testability and maintainability\n"
                "Return a JSON object with keys:\n"
                "- 'validation_results': List of validation checks\n"
                "- 'recommendations': Suggested improvements\n"
                "- 'code_quality_score': Overall quality score"
            ),
            expected_output=(
                "A JSON object containing validation results, recommendations, and a quality score."
            ),
            agent=self.code_agent(),
            context=[self.generate_code_task()]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.code_agent()
            ],
            tasks=[
                self.generate_code_task(),
                self.validate_code_task()
            ],
            process=Process.sequential
        )