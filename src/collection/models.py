import os
from dotenv import load_dotenv
import anthropic
import google.genai as genai
from openai import OpenAI

load_dotenv()


def call_gpt(prompt: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=1.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def call_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        temperature=1.0,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="models/gemini-2.5-pro",
        contents=prompt,
    )
    return response.text.strip()


MODEL_CALLERS = {
    "gpt": call_gpt,
    "claude": call_claude,
    "gemini": call_gemini,
}
