import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GNANI_API_KEY")

if api_key and api_key != "your_actual_gnani_api_key_here":
    print(" Success! GNANI_API_KEY successfully loaded from .env")
    print(f"Key preview: {api_key[:4]}...{api_key[-4:]}")
else:
    print(" Failed! GNANI_API_KEY not found or using default placeholder.")