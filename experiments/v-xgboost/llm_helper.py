import os
import time
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path

# Load .env from repo root
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

from azure.identity import DefaultAzureCredential, AzureCliCredential  # type: ignore
from azure.core.credentials import AzureKeyCredential  # type: ignore
import requests  # type: ignore

class AzureOpenAIError(Exception):
    pass

def generate_message(prediction_label: str, probability: Optional[float], lat: float, lng: float, datetime_str: str, deployment: Optional[str] = None, max_tokens: int = 256, temperature: float = 0.2) -> str:
    prob_text = f" with estimated confidence {probability:.2f}" if probability is not None else ""
    base = (
        f"Prediction: {prediction_label}{prob_text}. Location: (lat: {lat}, lng: {lng}). Date/time: {datetime_str}."
    )

    prompt = f"{base}\n\nPlease summarize what this prediction means in one short paragraph and list up to 3 practical actions a nearby person could take now."

    # Fallback to REST implementation if SDK isn't available
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

    # # print variables for debugging
    # print(f"DEBUG: endpoint={endpoint}, deployment_name={deployment_name}, key={'set' if key else 'not set'}, api_version={api_version}")

    if not (endpoint and deployment_name and key):
        raise AzureOpenAIError("Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, and AZURE_OPENAI_KEY. Put your .env file in the experiments folder.")

    url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {"Content-Type": "application/json", "api-key": key}
    body = {
        "messages": [
            {"role": "system", "content": "You are a concise safety assistant that summarizes crime predictions and gives practical, low-effort advice for citizens and authorities."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    backoff = 1.0
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=15)
        except requests.RequestException as e:
            if attempt == 2:
                raise AzureOpenAIError(f"Request failed: {e}")
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 200:
            data = resp.json()
            try:
                return data["choices"][0]["message"]["content"].strip()
            except Exception:
                raise AzureOpenAIError("Unexpected response format from Azure OpenAI")
        else:
            if attempt == 2:
                raise AzureOpenAIError(f"Azure OpenAI returned {resp.status_code}: {resp.text}")
            time.sleep(backoff)
            backoff *= 2
