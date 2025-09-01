from crewai import LLM
import os

# Gemini 1.5 Pro 	rpm2 	tpm32,000 	rpd50
gemini_llm =  LLM(
    model='gemini/gemini-1.5-flash',
    api_key=os.environ["GEMINI_API_KEY"]
            )



#mai-ds-r1
# Context
# 164K
# Max Output
# 164K
# Input
# $0
# Output
# $0
# Latency
# 13.19s
# Throughput
# 105.0t/s

microsoft_mai_ds_r1_llm =  LLM(
    model="microsoft/mai-ds-r1:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

deepseek_llm = llm = LLM(
    model="openrouter/deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

llama_llm = LLM(
    model="nvidia_nim/meta/llama3-70b-instruct",
    temperature=0.7,
    api_key=os.environ["NVIDIA_API_KEY"]
)

#pytest sdlc_ai_project/tests/test_requirement_analyst.py -v -s