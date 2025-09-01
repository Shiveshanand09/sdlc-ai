# from sdlc_ai_project.agents.requirements import RequirementAnalyzer
# from sdlc_ai_project.utils import collect_task_outputs
# from crewai import LLM
# from dotenv import load_dotenv
# import os

# load_dotenv()


# llm =  LLM(
#     model='gemini/gemini-1.5-flash',
#     api_key=os.environ["GEMINI_API_KEY"]
#             )

# requirement_analyst = RequirementAnalyzer(
#         user_requirements="Build a todo app with user authentication",
#         project_context="A simple web application for task management",
#         llm=llm,
#     )
# # req_crew = requirement_analyst.crew()
# # req_crew.kickoff()


# outputs = collect_task_outputs(requirement_analyst)