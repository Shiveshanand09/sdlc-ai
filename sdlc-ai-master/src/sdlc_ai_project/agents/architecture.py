import os
import json
import google.generativeai as genai
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

@CrewBase
class ArchitectureDesignAgent:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    """
    Architecture Design Agent:
    This agent generates detailed architecture representations based on analyzed requirements.
    It performs the following steps:
      1. Generates a detailed plain-text flowchart outlining the development process.
      2. Produces a comprehensive plain-text component diagram that defines system components and their interactions.
      3. Creates a structured blueprint of system components.
      4. Generates a high-level code skeleton to serve as a starting point for implementation.
      5. Decomposes the architecture into logical modules.
      6. Designs a data model that underpins the system.
      7. Specifies interfaces for module communication.
      8. Analyzes security and performance aspects.
      9. Provides a comprehensive technical roadmap.
    All outputs are formatted as JSON objects.
    """
    
    def __init__(self, requirement_analysis: str, tasks: str, project_context: str, tech_stack: str, extraction_task: str, llm):
        self.requirement_analysis = requirement_analysis
        self.tasks = tasks
        self.project_context = project_context
        self.tech_stack = tech_stack
        self.extraction_task = extraction_task
        self.llm = llm

    @agent
    def architecture_agent(self) -> Agent:
        return Agent(
            role=f"System Architecture Design Specialist for {self.project_context}",
            goal=(
                "Design a comprehensive system architecture that meets the project requirements while incorporating "
                "industry best practices and modern design patterns. Research similar systems to inform the design "
                "and ensure scalability, maintainability, and optimal performance."
            ),
            backstory=(
                "You are an expert system architect with extensive experience in designing scalable, maintainable systems. "
                "You excel at creating architectures that balance performance, security, and development efficiency. "
                "You have a deep understanding of modern architectural patterns and best practices across various domains."
            ),
            description=(
                "Designs comprehensive system architectures based on project requirements, focusing on scalability, "
                "maintainability, and optimal performance. Researches similar systems and incorporates best practices."
            ),
            llm=self.llm,
            # tools=[
            #     SerperDevTool(),  # For searching similar systems
            #     WebScraperTool(),  # For analyzing similar implementations
            #     CodeAnalysisTool(),  # For analyzing code patterns
            #     DocumentationTool(),  # For analyzing project documentation
            #     DirectoryReadTool(),  # For reading project files
            #     FileReadTool()  # For reading specific files
            # ]
        )

    @task
    def system_research_task(self) -> Task:
        return Task(
            description=(
                f"Research similar systems and architectures related to {self.project_context}. "
                "Focus on:\n"
                "- Similar projects in the same domain\n"
                "- Common architectural patterns and frameworks\n"
                "- Industry best practices and standards\n"
                "- Success stories and lessons learned\n"
                "- Performance optimization techniques\n"
                "Return a JSON object with keys:\n"
                "- 'similar_systems': List of relevant systems\n"
                "- 'architectural_patterns': Common patterns found\n"
                "- 'best_practices': Industry best practices\n"
                "- 'lessons_learned': Key insights from research\n"
                "- 'performance_optimizations': Optimization techniques"
            ),
            expected_output=(
                "A JSON object containing research findings about similar systems and architectures."
            ),
            agent=self.architecture_agent()
        )

    @task
    def system_flowchart_task(self) -> Task:
        return Task(
            description=(
                f"Analyze the requirements {self.requirement_analysis} and research findings to create a detailed system flowchart. "
                "Include:\n"
                "- System components and their interactions\n"
                "- Data flow and processing paths\n"
                "- User interaction points\n"
                "- External system integrations\n"
                "- Error handling and recovery paths\n"
                "- Performance critical paths\n"
                "Return JSON with 'flowchart' key containing the diagram and 'system_workflow' key "
                "detailing component interactions at each step."
            ),
            expected_output=(
                "JSON object containing a detailed system flowchart and component interaction specifications."
            ),
            agent=self.architecture_agent(),
            context=[self.system_research_task()]
        )

    @task
    def component_diagram_task(self) -> Task:
        return Task(
            description=(
                f"Based on the requirements: {self.requirement_analysis}, project context: {self.project_context}, "
                f"tech stack: {self.tech_stack}, research findings, and the system flowchart, create a comprehensive component diagram. "
                "Include:\n"
                "- System components and their responsibilities\n"
                "- Component interfaces and communication protocols\n"
                "- Data storage and processing components\n"
                "- External service integrations\n"
                "- Security and authentication components\n"
                "- Monitoring and logging components\n"
                "Return a JSON object with keys 'component_diagram' for the diagram and 'component_specifications' "
                "detailing component responsibilities and interfaces."
            ),
            expected_output=(
                "A JSON object containing a detailed component diagram and component specifications."
            ),
            agent=self.architecture_agent(),
            context=[self.system_flowchart_task(), self.system_research_task()]
        )

    @task
    def architecture_blueprint_task(self) -> Task:
        return Task(
            description=(
                "Create a detailed architecture blueprint based on the requirements, research findings, flowchart, and component diagram. "
                "For each component, specify:\n"
                "- Detailed responsibilities and capabilities\n"
                "- Interface specifications\n"
                "- Data models and schemas\n"
                "- Integration points and protocols\n"
                "- Security requirements\n"
                "- Performance considerations\n"
                "- Scalability strategies\n"
                "Return a JSON object with keys 'architecture_blueprint' for the detailed design and "
                "'implementation_guidelines' detailing development considerations."
            ),
            expected_output=(
                "A JSON object containing a detailed architecture blueprint and implementation guidelines."
            ),
            agent=self.architecture_agent(),
            context=[self.system_flowchart_task(), self.component_diagram_task(), self.system_research_task()]
        )

    @task
    def architecture_validation_task(self) -> Task:
        return Task(
            description=(
                "Validate the architecture design against requirements and research findings. Ensure:\n"
                "- All requirements are addressed\n"
                "- Alignment with similar successful systems\n"
                "- Incorporation of industry best practices\n"
                "- Proper security and performance considerations\n"
                "- Scalability and maintainability\n"
                "- Integration feasibility\n"
                "Return a JSON object with keys:\n"
                "- 'validation_results': List of validation checks\n"
                "- 'recommendations': Suggested improvements\n"
                "- 'best_practice_alignment': Alignment with research\n"
                "- 'risk_assessment': Potential risks and mitigations"
            ),
            expected_output=(
                "A JSON object containing architecture validation results and recommendations."
            ),
            agent=self.architecture_agent(),
            context=[self.architecture_blueprint_task(), self.system_research_task()]
        )

    # -------------------------------
    # Crew Integration
    # -------------------------------
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.architecture_agent()
            ],
            tasks=[
                self.system_research_task(),
                self.system_flowchart_task(),
                self.component_diagram_task(),
                self.architecture_blueprint_task(),
                self.architecture_validation_task()
            ],
            process=Process.sequential
        )
