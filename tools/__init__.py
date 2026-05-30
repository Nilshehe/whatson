# tools/__init__.py
from tools.web_search    import web_search
from tools.calculator    import calculator
from tools.code_executor import execute_python

# Convenience list — pass to create_react_agent or bind_tools
ALL_TOOLS         = [web_search, calculator, execute_python]
RESEARCH_TOOLS    = [web_search]
CODE_TOOLS        = [execute_python]
DATA_TOOLS        = [calculator]
