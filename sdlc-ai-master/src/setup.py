from setuptools import setup, find_packages

setup(
    name="sdlc_ai_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "crewai>=0.105.0",
        "crewai-tools>=0.40.1",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "cachetools>=5.3.2",
        "typing-extensions>=4.9.0",
        "typing-inspect>=0.9.0",
        "mypy-extensions>=1.0.0",
        "python-dateutil>=2.8.2",
        "pathlib>=1.0.1",
    ],
    python_requires=">=3.8",
) 


#source venv/bin/activate && pip install -e . && python -m sdlc_ai_project.main