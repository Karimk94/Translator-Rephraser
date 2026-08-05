import os
import re
import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables from a .env file
load_dotenv()

# --- GovAI LLM API Configuration ---
GOVAI_API_URL = os.getenv("GOVAI_API_URL")
GOVAI_API_KEY = os.getenv("GOVAI_API_KEY")
GOVAI_MODEL = os.getenv("GOVAI_MODEL")

# Initialize the FastAPI application
app = FastAPI(title="AI Writing Assistant API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Template configuration for serving index.html
templates = Jinja2Templates(directory="templates")


# --- Request Model ---
class GenerateRequest(BaseModel):
    text: str
    task: str = "translate"


def is_arabic(text):
    """Detects if the text contains a significant number of Arabic characters."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search('[\u0600-\u06FF]', text))

def clean_response(text):
    """
    Removes all Arabic diacritics (Tashkeel) and stray question marks.
    """
    if not text:
        return text
    
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    cleaned_text = re.sub(arabic_diacritics, '', text)
    cleaned_text = cleaned_text.replace('?', '')
    
    return cleaned_text

def create_messages(text, task, source_lang):
    """
    Constructs the system and user messages based on the selected task.
    Returns a list of message dicts for the OpenAI-compatible chat API.
    """
    if task == 'translate':
        target_lang = "Arabic" if source_lang == "English" else "English"
        
        diacritics_rule = ""
        other_rules = ""

        if target_lang == "Arabic":
            diacritics_rule = """1. **ABSOLUTELY CRITICAL**: Your response MUST NOT contain any Arabic diacritics (Tashkeel / formations).
   - **Example (Good)**: "مرحبا"
2. **CRITICAL**: Do NOT output any question mark `?` characters."""
            other_rules = f"""3. Your response MUST contain ONLY the translated text.
4. Do NOT add any comments, explanations, or introductory phrases.
5. **CRITICAL**: You MUST convert every single {source_lang} word into {target_lang}."""
        else:
            other_rules = f"""1. Your response MUST contain ONLY the translated text.
2. Do NOT add any comments.
3. **CRITICAL**: You MUST convert every single {source_lang} word into {target_lang}."""

        system_msg = f"You are a direct translation engine. Translate the provided {source_lang} text to {target_lang}."
        user_msg = f"""Follow these rules exactly:
{diacritics_rule}
{other_rules}

### TASK ###
{source_lang} Text: "{text}" """
        
    elif task == 'tokenize':
        tag_lang = "Arabic" if source_lang == "Arabic" else "English"
        json_key = "arabic_tags" if source_lang == "Arabic" else "english_tags"
        
        system_msg = "You are a JSON output machine. Your only function is to output a specific JSON structure."
        user_msg = f"""Follow these steps exactly:

1.  Analyze this {source_lang} text: "{text}"
2.  Extract the key nouns, entities, and concepts for use as search tags.
3.  **All tags MUST be in {tag_lang}.** Do not mix languages.
4.  **Correct any spelling errors and standardize abbreviations.**
5.  Remove all stop words, non-essential words, and duplicate entries.
6.  If generating {tag_lang} tags, do NOT include any diacritics (Tashkeel / formations).
7.  Output **NOTHING** except for the completed JSON structure below. Do not use markdown.

COPY AND PASTE THIS TEMPLATE, THEN FILL IT IN:
{{"{json_key}": []}}

Your entire response must be only the filled-out template."""

    elif task == 'rephrase':
        system_msg = f"You are a professional editor. Your task is to rewrite text to be clearer, more concise, and more professional, while strictly preserving the original meaning."
        user_msg = f"""Rules:
1. Do NOT change the language. Keep it in {source_lang}.
2. Output ONLY the rephrased text. No conversational filler.
3. If the text is Arabic, do NOT use diacritics (Tashkeel).

Text: "{text}" """

    else:
        system_msg = "You are a helpful assistant."
        user_msg = text

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


@app.post("/generate")
async def generate(req: GenerateRequest):
    original_text = req.text
    task = req.task
    source_language = "Arabic" if is_arabic(original_text) else "English"
    messages = create_messages(original_text, task, source_language)
    
    payload = {
        "model": GOVAI_MODEL,
        "messages": messages,
        "stream": True,
        "max_tokens": 4096,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GOVAI_API_KEY}",
    }

    print(f"DEBUG: Task: {task} | Source: {source_language} | Model: {GOVAI_MODEL}")

    def generator():
        try:
            response = requests.post(
                GOVAI_API_URL, headers=headers, json=payload, stream=True, timeout=120
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    
                    # SSE format: lines start with "data: "
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        
                        # End of stream signal from OpenAI-compatible API
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            # Extract content delta from OpenAI-compatible streaming format
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    cleaned = clean_response(content)
                                    yield f"data: {cleaned}\n\n"
                        except json.JSONDecodeError:
                            continue

            yield "data: [END_OF_STREAM]\n\n"

        except requests.exceptions.RequestException as e:
            print(f"Error during generation: {e}")
            yield "data: [ERROR]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renders the main page."""
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    port = int(os.environ.get("HTTP_PLATFORM_PORT", 5005))
    print(f"Starting Translator Rephraser API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)