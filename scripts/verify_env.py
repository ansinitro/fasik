import os
from dotenv import load_dotenv

def verify():
    print("Testing environment variable loading...")
    
    # 1. Check if .env exists
    if os.path.exists(".env"):
        print("✅ .env file found")
    else:
        print("❌ .env file NOT found")
        return

    # 2. Load environment variables
    load_dotenv()
    
    # 3. Check API Key
    api_key = os.getenv("FACEIT_API_KEY")
    if api_key:
        masked_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        print(f"✅ FACEIT_API_KEY loaded successfully: {masked_key}")
    else:
        print("❌ FACEIT_API_KEY not found in environment variables")

if __name__ == "__main__":
    verify()
