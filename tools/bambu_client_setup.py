from bambulab import BambuClient
from dotenv import load_dotenv
import os

load_dotenv()

client = BambuClient(token=os.getenv("BAMBU_TOKEN"))
devices = client.get_devices()
printer = devices[0]   # första skrivaren
device_id = printer["dev_id"]
