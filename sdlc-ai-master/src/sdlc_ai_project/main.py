from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import json
import re
from sdlc_ai_project.agents.knowledge import KnowledgeBaseAgent
from sdlc_ai_project.agents.requirements import RequirementAnalyzer
from sdlc_ai_project.agents.architecture import ArchitectureDesignAgent
from sdlc_ai_project.agents.skeletons import CodeSkeletonGenerator
from sdlc_ai_project.agents.generator import CodeGenerator
from sdlc_ai_project.llms import deepseek_llm, gemini_llm
from sdlc_ai_project.utils import collect_task_outputs, save_to_json
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="SDLC AI Agent API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# Request Models
# ------------------------
class BasicInput(BaseModel):
    user_requirements: str
    project_context: str
    title: str

class RequirementInput(BasicInput):
    knowledge_output: Optional[Dict[str, Any]] = None

class ArchitectureInput(BasicInput):
    requirement_output: Optional[Dict[str, Any]] = None
    knowledge_output: Optional[Dict[str, Any]] = None

class SkeletonInput(BaseModel):
    architecture_output: Dict[str, Any]
    project_context: str
    title: str

class CodeGenInput(SkeletonInput):
    skeleton_output: Dict[str, Any]

class ValidationInput(BaseModel):
    task_name: str
    modified_output: Any


def parse_json_from_markdown(text: Any) -> Any:
    """
    Parse JSON from markdown code blocks or direct JSON strings.
    
    If the input is already a dict/list, return it as is.
    If the input is a string containing ```json {...}```, extract and parse the JSON.
    Otherwise, try to parse the string as JSON directly.
    """
    # If already a dict/list, return as is
    if isinstance(text, (dict, list)):
        return text
        
    # If not a string, return as is
    if not isinstance(text, str):
        return text
        
    # Try to extract JSON from markdown code block
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            # If JSON parsing fails, return the extracted content as a string
            return match.group(1)
    
    # If no markdown pattern, try parsing as direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If all parsing fails, return the original input
        return text


def process_agent_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the agent output to ensure all values are properly formatted JSON objects.
    Recursively process nested dictionaries and lists.
    """
    if isinstance(output, dict):
        return {k: process_agent_output(parse_json_from_markdown(v)) for k, v in output.items()}
    elif isinstance(output, list):
        return [process_agent_output(parse_json_from_markdown(item)) for item in output]
    else:
        return parse_json_from_markdown(output)


# Override the default collect_task_outputs function to avoid console input
def non_interactive_collect_outputs(agent) -> Dict[str, Any]:
    """Collect outputs from agent tasks without user interaction."""
    agent_crew = agent.crew()
    agent_crew.kickoff()
    
    outputs = {}
    for task in agent_crew.tasks:
        task_name = task.name
        output = task.output.raw
        # Parse any JSON in markdown code blocks
        parsed_output = parse_json_from_markdown(output)
        outputs[task_name] = parsed_output
    
    # Process the entire output structure to ensure proper JSON formatting
    return process_agent_output(outputs)

# ------------------------
# Agent Endpoints
# ------------------------

@app.post("/agent/knowledge")
async def run_knowledge_agent(request: BasicInput):
    try:
        llm = gemini_llm
        agent = KnowledgeBaseAgent(user_requirements=request.user_requirements, project_context=request.project_context, llm=llm)
        output = non_interactive_collect_outputs(agent)
        save_to_json("Knowledge", output, request.title)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/requirements")
async def run_requirement_analyzer(request: BasicInput):
    try:
        llm = gemini_llm
        agent = RequirementAnalyzer(user_requirements=request.user_requirements, project_context=request.project_context, llm=llm)
        output = non_interactive_collect_outputs(agent)
        save_to_json("Requirements", output, request.title)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/architecture")
async def run_architecture_agent(request: ArchitectureInput):
    try:
        llm = gemini_llm
        agent = ArchitectureDesignAgent(
            requirement_analysis=request.requirement_output,
            tasks=request.requirement_output["task_generation_task"],
            project_context=request.project_context,
            tech_stack=request.knowledge_output["finalize_tech_stack"],
            extraction_task=request.requirement_output["extraction_task"],
            llm=llm
        )
        output = non_interactive_collect_outputs(agent)
        save_to_json("Architecture", output, request.title)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/skeleton")
async def run_skeleton_generator(request: SkeletonInput):
    try:
        llm = gemini_llm
        agent = CodeSkeletonGenerator(
            architecture_design=request.architecture_output,
            project_context=request.project_context,
            llm=llm
        )
        output = non_interactive_collect_outputs(agent)
        save_to_json("Skeletons", output, request.title)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/codegen")
async def run_code_generator(request: CodeGenInput):
    try:
        llm = gemini_llm
        agent = CodeGenerator(
            architecture_design=request.architecture_output,
            project_context=request.project_context,
            skeletons=request.skeleton_output["code_skeleton_task"],
            module_boilerplate=request.skeleton_output["module_boilerplate_task"],
            llm=llm
        )
        output = non_interactive_collect_outputs(agent)
        save_to_json("Generator", output, request.title)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/validate")
async def validate_task_output(request: ValidationInput):
    try:
        # Log the modified output
        print(f"Modified Output for {request.task_name}: {json.dumps(request.modified_output, indent=2)}")
        
        # Save the modified output to a temporary file for debugging
        os.makedirs("validation_logs", exist_ok=True)
        with open(f"validation_logs/{request.task_name}_modified.json", "w") as f:
            json.dump(request.modified_output, f, indent=2)
        
        return {"message": f"Output for {request.task_name} validated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))