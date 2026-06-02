import os
from uuid import uuid4
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

GIGACHAT_AUTH_KEY = os.getenv("GIGACHAT_AUTH_KEY")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_BASE_URL = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
GIGACHAT_AUTH_URL = os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "true").lower() == "true"

print("=== GigaChat Diagnostic ===")
print(f"Provider: gigachat")
print(f"Auth URL: {GIGACHAT_AUTH_URL}")
print(f"Base URL: {GIGACHAT_BASE_URL}")
print(f"Model: {GIGACHAT_MODEL}")
print(f"Verify SSL: {GIGACHAT_VERIFY_SSL}")
print()

if not GIGACHAT_AUTH_KEY:
    print("ERROR: GIGACHAT_AUTH_KEY is missing in .env")
    exit(1)

print("Step 1: Getting OAuth token...")
try:
    response = requests.post(
        GIGACHAT_AUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid4()),
            "Authorization": f"Basic {GIGACHAT_AUTH_KEY}",
        },
        data={"scope": GIGACHAT_SCOPE},
        timeout=30,
        verify=GIGACHAT_VERIFY_SSL,
    )
    oauth_status = response.status_code
    print(f"OAuth status: {oauth_status}")

    if oauth_status != 200:
        print(f"OAuth FAILED: {response.text[:200]}")
        exit(1)

    token = response.json()["access_token"]
    print("OAuth SUCCESS: token obtained")
except requests.exceptions.SSLError as e:
    print(f"SSL ERROR: {e}")
    print("Try setting GIGACHAT_VERIFY_SSL=false in .env")
    exit(1)
except Exception as e:
    print(f"OAuth ERROR: {e}")
    exit(1)

print()
print("Step 2: Getting available models...")
try:
    models_url = f"{GIGACHAT_BASE_URL.rstrip('/')}/models"
    response = requests.get(
        models_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        timeout=30,
        verify=GIGACHAT_VERIFY_SSL,
    )
    models_status = response.status_code
    print(f"Models status: {models_status}")

    if models_status != 200:
        print(f"Models request FAILED: {response.text[:200]}")
    else:
        models = response.json()
        print("Models SUCCESS:")
        if "data" in models:
            for m in models["data"]:
                print(f"  - id: {m.get('id')}, object: {m.get('object')}")
        else:
            print(f"  {models}")
except requests.exceptions.SSLError as e:
    print(f"SSL ERROR: {e}")
except Exception as e:
    print(f"Models ERROR: {e}")

print()
print("=== Diagnostic Complete ===")