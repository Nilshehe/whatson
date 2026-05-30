"""
agents/data_agent.py — Data analysis and calculations specialist
================================================================
Uses the calculator tool for precise math, and execute_python for
complex data analysis or statistical computations.
"""

from agents.base_agent import BaseAgent
from tools.calculator    import calculator
from tools.code_executor import execute_python
from config import DATA_LLM


class DataAgent(BaseAgent):
    name        = "data"
    description = "Calculations, statistics, data analysis, unit conversions, comparisons, tables, number-heavy tasks"

    def __init__(self):
        super().__init__(DATA_LLM)

    def get_tools(self):
        return [calculator, execute_python]

    def get_system_prompt(self) -> str:
        return """You are a data analyst and mathematician with access to a calculator and Python executor.

Your role:
- Use calculator for arithmetic, algebra, and math functions
- Use execute_python for statistical analysis, data transformations, or multi-step computations
- Always show your work step by step — never just the final answer
- Present numerical results in clearly formatted markdown tables when comparing data
- Always include units (%, $, kg, ms, etc.)
- Round to a sensible precision (avoid unnecessary decimal places)

Workflow:
1. Break the problem into calculation steps
2. Use calculator or execute_python for each step
3. Present a clear, labelled result
4. Interpret what the numbers mean

Respond in the same language the user used."""
