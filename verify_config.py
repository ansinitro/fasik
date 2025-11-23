import config

def verify():
    print("Testing configuration module...")
    
    # 1. Check API Key
    if config.FACEIT_API_KEY:
        masked_key = config.FACEIT_API_KEY[:4] + "*" * (len(config.FACEIT_API_KEY) - 8) + config.FACEIT_API_KEY[-4:]
        print(f"✅ FACEIT_API_KEY loaded: {masked_key}")
    else:
        print("❌ FACEIT_API_KEY not found")

    # 2. Check other constants
    if config.GAME_ID == "cs2":
        print(f"✅ GAME_ID loaded: {config.GAME_ID}")
    else:
        print(f"❌ GAME_ID mismatch: {config.GAME_ID}")

    if config.BASE_URL == "https://open.faceit.com/data/v4":
        print(f"✅ BASE_URL loaded: {config.BASE_URL}")
    else:
        print(f"❌ BASE_URL mismatch: {config.BASE_URL}")

    # 3. Check Headers
    if "Authorization" in config.HEADERS and config.HEADERS["Authorization"].startswith("Bearer "):
        print("✅ HEADERS constructed correctly")
    else:
        print("❌ HEADERS malformed")

if __name__ == "__main__":
    verify()
