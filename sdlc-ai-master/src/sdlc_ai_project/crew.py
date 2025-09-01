from crewai import Crew, Agent, Task
from .tools_old import get_tool
import yaml
import os

def load_crew():
    # Load agents from YAML
    with open("sdlc_ai_project/config/agents.yaml", "r") as f:
        agents_config = yaml.safe_load(f)
    
    # Create agents with their IDs stored
    agents = {}
    for agent_id, config in agents_config.items():
        agent = Agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=[get_tool(tool_name) for tool_name in config.get("tools", [])]
        )
        agents[agent_id] = agent
    
    # Load tasks from YAML
    with open("sdlc_ai_project/config/tasks.yaml", "r") as f:
        tasks_config = yaml.safe_load(f)
    
    # Create tasks
    tasks = []
    for task_id, config in tasks_config.items():
        agent_id = config["agent"]
        if agent_id not in agents:
            raise ValueError(f"Agent '{agent_id}' not found for task '{task_id}'")
            
        task = Task(
            description=config["description"],
            agent=agents[agent_id],
            expected_output=config.get("expected_output", "Task completed successfully")
        )
        tasks.append(task)
    
    # Convert agents dict to list for Crew initialization
    agents_list = list(agents.values())
    
    return Crew(agents=agents_list, tasks=tasks)
