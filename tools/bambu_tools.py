from langchain.tools import tool
from bambulab import BambuClient
from dotenv import load_dotenv
import os

load_dotenv()

# Initiera Cloud API-klienten
client = BambuClient(token=os.getenv("BAMBU_TOKEN"))

# Hämta första skrivaren
devices = client.get_devices()
printer = devices[0]
device_id = printer["dev_id"]


# =========================
# STATUS
# =========================

@tool
def get_printer_status() -> str:
    """Get printer status, progress and ETA."""
    status = client.get_print_status(force=True)

    dev = next((d for d in status.get("devices", []) if d["dev_id"] == device_id), None)
    if not dev:
        return "No status available"

    return (
        f"State: {dev.get('print_status')}\n"
        f"Progress: {dev.get('progress')}%\n"
        f"ETA: {dev.get('eta')}\n"
        f"Nozzle: {dev.get('nozzle_temp')}°C\n"
        f"Bed: {dev.get('bed_temp')}°C"
    )


@tool
def get_temperatures() -> str:
    """Get nozzle and bed temperatures."""
    status = client.get_print_status(force=True)
    dev = next((d for d in status.get("devices", []) if d["dev_id"] == device_id), None)
    if not dev:
        return "No temperature data"

    return (
        f"Nozzle: {dev.get('nozzle_temp')}°C\n"
        f"Bed: {dev.get('bed_temp')}°C"
    )


# =========================
# PRINT CONTROL
# =========================

@tool
def start_print(filename: str) -> str:
    """Start a print from cloud storage."""
    try:
        result = client.start_cloud_print(device_id=device_id, filename=filename)
        return f"Print started: {result}"
    except Exception as e:
        return f"Failed: {e}"


@tool
def pause_print() -> str:
    """Pause active print."""
    return "Pause not supported in Cloud API"


@tool
def resume_print() -> str:
    """Resume paused print."""
    return "Resume not supported in Cloud API"


@tool
def cancel_print() -> str:
    """Cancel active print."""
    return "Cancel not supported in Cloud API"


# =========================
# AMS
# =========================

@tool
def list_filaments() -> str:
    """Show AMS filament information."""
    ams = client.get_ams_filaments(device_id)
    return str(ams)


@tool
def get_active_filament() -> str:
    """Current filament slot."""
    ams = client.get_ams_filaments(device_id)
    if not ams.get("ams_units"):
        return "No AMS installed"
    return str(ams["ams_units"][0]["trays"])


# =========================
# CAMERA
# =========================

@tool
def get_camera_snapshot() -> str:
    """Get cloud camera URLs and TTCode."""
    cam = client.get_camera_urls(device_id)
    return str(cam)


# =========================
# FILES
# =========================

@tool
def list_printer_files() -> str:
    """List cloud files."""
    files = client.get_cloud_files()
    if not files:
        return "No cloud files found"
    return "\n".join(f.get("name", "unknown") for f in files)


@tool
def delete_file(filename: str) -> str:
    """Delete file from cloud."""
    return "Cloud API does not support deleting files"


# =========================
# DIAGNOSTICS
# =========================

@tool
def get_errors() -> str:
    """Get printer error codes."""
    status = client.get_print_status(force=True)
    dev = next((d for d in status.get("devices", []) if d["dev_id"] == device_id), None)
    if not dev:
        return "No error data"
    return str(dev.get("error"))


@tool
def get_full_state() -> str:
    """Return complete raw printer state."""
    return str(client.get_print_status(force=True))
