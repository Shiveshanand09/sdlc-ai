from sdlc_ai_project.utils import run_sdlc, save_to_json
from sdlc_ai_project.llms import deepseek_llm, gemini_llm

def main():

    run_sdlc(user_requirements="Build a todo app with user authentication",project_context="A simple web application for task management", llm=gemini_llm, title='todo')

    pass

if __name__ == "__main__":
    main()




# from sdlc_ai_project.crew import load_crew
# import yaml
# from crewai import Agent, LLM
# import os
# from dotenv import load_dotenv

# def load_config():
#     with open("sdlc_ai_project/config/config.yaml", "r") as f:
#         return yaml.safe_load(f)

# if __name__ == "__main__":
#     # Load environment variables
#     load_dotenv()
    
#     # Load configuration
#     config = load_config()
    
#     # Get the Gemini API key from environment variable
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("GEMINI_API_KEY environment variable is not set")
    
#     # Configure the LLM
#     llm =  LLM(
#               model='gemini/gemini-1.5-flash',
#               api_key=os.environ["GEMINI_API_KEY"]
#             )
    
#     Agent.llm = llm
#     # Set the LLM for CrewAI
#     Agent.default_llm = llm
    
#     # Load and run the crew
#     crew = load_crew()
#     results = crew.kickoff(inputs={
#         "user_requirements": "Build a scalable todo app with auth, due dates, and cloud sync."
#     })
#     print(results)
