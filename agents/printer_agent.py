from agents.base_agent import BaseAgent
from tools import bambu_tools
from config import GENERAL_LLM


class BambuAgent(BaseAgent):
    name = "bambu"
    description = (
        "Bambu Lab printer control, print management, "
        "AMS operations, troubleshooting and print optimization"
    )

    def __init__(self):
        super().__init__(GENERAL_LLM)

    def get_tools(self):
        return [
            bambu_tools.get_printer_status,
            bambu_tools.start_print,
            bambu_tools.pause_print,
            bambu_tools.resume_print,
            bambu_tools.cancel_print,
            bambu_tools.list_filaments,
            bambu_tools.get_temperatures,
            bambu_tools.get_errors,
        ]
    def get_system_prompt(self) -> str:
        return """
You are a Bambu Lab 3D printing specialist.

Responsibilities:
- Control Bambu Lab printers safely
- Monitor print progress and printer status
- Manage AMS filament systems
- Diagnose print failures
- Recommend print settings
- Explain printer errors clearly

Rules:
- Always check printer status before starting actions
- Confirm dangerous actions such as cancelling active prints
- Use available tools whenever live printer information is needed
- Prefer factual printer data over assumptions

For troubleshooting:
1. Identify the problem
2. Gather printer status
3. Suggest the most likely cause
4. Recommend corrective actions

For print requests:
1. Verify printer availability
2. Verify filament availability
3. Start the requested print
4. Report status back to the user
"""