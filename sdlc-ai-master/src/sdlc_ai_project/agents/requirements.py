import os
import json
import google.generativeai as genai
from crewai import Crew, Agent, Task, Process
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from sdlc_ai_project.tools import code_research_tool
#from crewai_tools import SerperDevTool, WebScraperTool, CodeAnalysisTool, DocumentationTool
load_dotenv()

@CrewBase
class RequirementAnalyzer:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def __init__(self, user_requirements: str, project_context: str, llm):
        self.user_requirements = user_requirements
        self.project_context = project_context
        self.llm = llm

    @agent
    def requirement_agent(self) -> Agent:
        return Agent(
            role=f"AI-Driven Requirements Engineering Specialist for {self.project_context}",
            goal=(
                "Leverage advanced NLP and machine learning techniques to analyze and structure software requirements. "
                "Transform complex user inputs into actionable development tasks while identifying implicit needs and potential automation opportunities. "
                "Prioritize tasks to optimize AI agent collaboration and automation potential."
            ),
            backstory=(
                "You are an expert in AI-driven requirements engineering, specializing in identifying automation opportunities "
                "and structuring requirements for AI agent collaboration. You excel at uncovering implicit needs and "
                "transforming them into clear, actionable development tasks."
            ),
            description=(
                "Utilizes advanced NLP to extract and structure requirements, identifying automation opportunities "
                "and optimizing task flow for AI agent collaboration."
            ),
            llm=self.llm,
            knowledge_sources=[],
            # tools=[code_research_tool]
        )

    @agent
    def intent_analysis_agent(self) -> Agent:
        return Agent(
            role="User Intent Analysis Specialist",
            goal=(
                "Deeply analyze user requirements to understand the underlying intent, business goals, "
                "and implicit needs. Research similar projects and industry best practices to inform "
                "task creation and automation opportunities."
            ),
            backstory=(
                "You are an expert in requirements analysis and user intent understanding. "
                "You excel at uncovering implicit needs and transforming vague requirements "
                "into clear, actionable development goals. You have extensive experience "
                "researching similar projects and identifying successful patterns."
            ),
            description=(
                "Analyzes user requirements to understand true intent and business goals, "
                "researches similar projects, and identifies automation opportunities."
            ),
            llm=self.llm,
            # tools=[
            #     SerperDevTool(),  # For searching similar projects and best practices
            #     WebScraperTool(),  # For analyzing similar project implementations
            #     CodeAnalysisTool(),  # For analyzing code patterns
            #     DocumentationTool()  # For analyzing project documentation
            # ]
            # tools=[code_research_tool]
        )

    @task
    def extraction_task(self) -> Task:
        return Task(
            description=(
                f"Analyze the user requirements: {self.user_requirements} and project context: {self.project_context} "
                "to extract essential functionalities, constraints, and automation opportunities. "
                "Identify tasks that can be automated by AI agents and those requiring human intervention. "
                "Return a JSON object with a key 'tasks' mapping to a list of objects, each containing: "
                "'task_id', 'description', 'priority' (low/med/high), 'automation_potential' (high/medium/low), "
                "'ai_agent_type' (if applicable), and 'dependencies'."
            ),
            expected_output=(
                "A JSON object containing prioritized development tasks with automation potential analysis, "
                "arranged in a logical execution order for AI agent collaboration."
            ),
            agent=self.requirement_agent(),
        )

    @task
    def project_research_task(self) -> Task:
        return Task(
            description=(
                f"Research similar projects and implementations related to {self.project_context}. "
                "Focus on:\n"
                "- Similar projects in the same domain\n"
                "- Common implementation patterns\n"
                "- Industry best practices\n"
                "- Success stories and lessons learned\n"
                "- Automation approaches used\n"
                "Return a JSON object with keys:\n"
                "- 'similar_projects': List of relevant projects\n"
                "- 'implementation_patterns': Common patterns found\n"
                "- 'best_practices': Industry best practices\n"
                "- 'lessons_learned': Key insights from research\n"
                "- 'automation_examples': Automation approaches used"
            ),
            expected_output=(
                "A JSON object containing research findings about similar projects and implementations."
            ),
            agent=self.intent_analysis_agent()
        )

    @task
    def intent_analysis_task(self) -> Task:
        return Task(
            description=(
                f"Analyze the user requirements: {self.user_requirements} and project context: {self.project_context} "
                "to understand the true intent and business goals. Consider:\n"
                "- What problem is the user trying to solve?\n"
                "- What are the implicit requirements?\n"
                "- What are the key success criteria?\n"
                "- How do similar projects address these needs?\n"
                "- What automation opportunities exist?\n"
                "Return a JSON object with keys:\n"
                "- 'business_goals': List of main business objectives\n"
                "- 'implicit_requirements': Unstated but important requirements\n"
                "- 'success_criteria': Key metrics for success\n"
                "- 'similar_project_insights': Insights from similar projects\n"
                "- 'automation_potential': Areas suitable for automation"
            ),
            expected_output=(
                "A JSON object containing a comprehensive analysis of user intent, "
                "business goals, and automation opportunities."
            ),
            agent=self.intent_analysis_agent(),
            context=[self.project_research_task()]
        )

    @task
    def task_generation_task(self) -> Task:
        return Task(
            description=(
                "Based on the user intent analysis and project research, generate a comprehensive list of tasks. "
                "Consider:\n"
                "- Similar project implementations\n"
                "- Industry best practices\n"
                "- Automation opportunities\n"
                "- Technical requirements\n"
                "Return a JSON object with keys:\n"
                "- 'tasks': List of development tasks\n"
                "- 'automation_strategies': Automation approaches\n"
                "- 'implementation_patterns': Recommended patterns\n"
                "- 'dependencies': Task dependencies\n"
                "- 'success_metrics': Success criteria"
            ),
            expected_output=(
                "A JSON object containing a comprehensive list of tasks based on research and analysis."
            ),
            agent=self.requirement_agent(),
            context=[self.intent_analysis_task(), self.project_research_task()]
        )

    @task
    def requirement_validation_task(self) -> Task:
        return Task(
            description=(
                "Validate the extracted requirements against the user intent analysis. Ensure:\n"
                "- All business goals are addressed\n"
                "- Implicit requirements are included\n"
                "- Success criteria are measurable\n"
                "- Technical constraints are considered\n"
                "- Automation opportunities are maximized\n"
                "Return a JSON object with keys:\n"
                "- 'validated_requirements': Updated requirements list\n"
                "- 'gaps_identified': Any missing requirements\n"
                "- 'automation_improvements': Enhanced automation opportunities\n"
                "- 'risk_assessment': Potential risks and mitigation strategies"
            ),
            expected_output=(
                "A JSON object containing validated requirements with identified gaps "
                "and enhanced automation opportunities."
            ),
            agent=self.requirement_agent(),
            context=[self.intent_analysis_task(), self.extraction_task()]
        )

    @task
    def subtask_breakdown_task(self) -> Task:
        return Task(
            description=(
                "Given the extracted high-level development tasks, decompose each task into atomic subtasks "
                "optimized for AI agent execution. For each subtask, specify: "
                "- Required AI agent capabilities\n"
                "- Input/output specifications\n"
                "- Success criteria\n"
                "- Potential automation challenges\n"
                "Return a JSON object with a key 'tasks' where each task maps to its list of subtasks, "
                "each containing 'subtask_id', 'description', 'priority', 'ai_agent_requirements', "
                "'input_spec', 'output_spec', and 'success_criteria'."
            ),
            expected_output=(
                "A JSON object with structured breakdown of subtasks optimized for AI agent execution, "
                "including detailed specifications for automation."
            ),
            agent=self.requirement_agent(),
            context=[self.extraction_task()]
        )


    # -------------------------------
    # Crew Integration
    # -------------------------------
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.intent_analysis_agent(),
                self.requirement_agent(),
            ],
            tasks=[
                self.extraction_task(),
                self.project_research_task(),
                self.intent_analysis_task(),
                self.task_generation_task(),
                self.requirement_validation_task(),
                self.subtask_breakdown_task(),
            ],
            process=Process.sequential
        )
