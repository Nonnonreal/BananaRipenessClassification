import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    print("\n" + "=" * 65)
    print("🍌 BananaVision AI Web Application")
    print(f"🚀 Server started at: http://{host}:{port}")
    print("=" * 65 + "\n")
    
    app.run(host=host, port=port, debug=debug)
