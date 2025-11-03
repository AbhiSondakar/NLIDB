import os
import streamlit as st
import requests
from sqlalchemy import inspect, Engine
from typing import TypedDict, Tuple, Optional
import json

# OpenRouter model registry - format: provider/model-name[:tag]
MODELS = {
    'deepseek/deepseek-33b-chat': {  # OpenRouter's exact model ID
        'description': 'DeepSeek Chat 33B - optimized for retrieval, semantic search, and structured generation',
        'supports_json': True,
        'max_tokens': 32768,
        'best_for': ['semantic_search', 'retrieval_augmented_generation', 'structured_output', 'sql']
    }
}

# Map from our model keys to OpenRouter's model IDs
OPENROUTER_MODEL_IDS = {
    'deepseek/deepseek-33b-chat': 'deepseek/deepseek-33b-chat',  # Exact match
}

# Default model for text-to-SQL tasks (use our internal key)
DEFAULT_MODEL = 'deepseek/deepseek-33b-chat'

def get_available_models() -> list[str]:
    """Get list of available models from the configured DeepSeek API.

    Falls back to the DEFAULT_MODEL if no API is configured or the request fails.
    """
    api_url = os.getenv("DEEPSEEK_API_URL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_url or not api_key:
        # No external API configured; return default only
        return [DEFAULT_MODEL]

    try:
        # Use the base API to call the /models endpoint. If the DEEPSEEK_API_URL includes
        # a /chat/completions suffix, strip it so we don't end up with /chat/completions/models.
        models_base = api_url.rstrip('/').split('/chat')[0]
        resp = requests.get(
            f"{models_base}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        # Expecting a list of model names under 'models' or top-level list
        if isinstance(data, dict) and "models" in data:
            return data["models"]
        if isinstance(data, list):
            return data
        # Unexpected format; return default
        return [DEFAULT_MODEL]
    except Exception as e:
        st.warning(f"Could not fetch available models from DeepSeek API: {e}")
        return [DEFAULT_MODEL]

def select_best_model() -> str:
    """
    Select the best available model for text-to-SQL tasks.
    Prioritizes models in this order:
    1. SQL/Code specialized models
    2. Ultra models (if available)
    3. Pro models
    4. Legacy models as fallback
    """
    try:
        available = get_available_models()
        
        # Priority order for SQL/Code tasks
        model_priority = [
            'deepseek/deepseek-33b-chat',      # OpenRouter's exact model ID for DeepSeek
        ]
        
        # Try each model in order of preference
        for model_name in model_priority:
            if model_name in available:
                # Log model capabilities
                if model_name in MODELS:
                    model_info = MODELS[model_name]
                    st.write(f"Selected model: {model_name}")
                    st.write(f"Capabilities: {model_info['description']}")
                    st.write(f"Best for: {', '.join(model_info['best_for'])}")
                return model_name
                
        return DEFAULT_MODEL
    except Exception as e:
        st.warning(f"Model selection error: {e}")
        return DEFAULT_MODEL

# Define the structured output we expect from the AI model using a TypedDict
class AISqlResponse(TypedDict):
    explanation: str
    sqlQuery: str

# Optional DeepSeek API configuration check (non-fatal)
# The app can run without an external API if you prefer to use a local/alternate AI service.
_deepseek_url = os.getenv("DEEPSEEK_API_URL")
_deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not _deepseek_url or not _deepseek_key:
    st.info("DeepSeek API not configured. Set DEEPSEEK_API_URL and DEEPSEEK_API_KEY to enable remote model calls.")

@st.cache_data(ttl=600)  # Cache schema for 10 minutes
def get_database_schema(_engine: Engine) -> str:
    """
    Inspects the database connected to by the engine and returns a simplified
    schema string (CREATE TABLE statements) for the LLM.
    Uses a leading underscore in the parameter name to tell Streamlit not to hash the engine.
    """
    try:
        inspector = inspect(_engine)
        schema_str = ""
        table_names = inspector.get_table_names()
        
        for table_name in table_names:
            schema_str += f"CREATE TABLE {table_name} (\n"
            columns = inspector.get_columns(table_name)
            for i, column in enumerate(columns):
                col_name = column['name']
                col_type = str(column['type'])
                schema_str += f"  {col_name} {col_type}"
                if i < len(columns) - 1:
                    schema_str += ",\n"
            schema_str += "\n);\n\n"
        
        if not schema_str:
            return "Error: No tables found in the database. The read-only user may not have proper permissions."
            
        return schema_str
    except Exception as e:
        return f"Error inspecting schema: {e}"

def get_ai_response(user_prompt: str, schema: str) -> Tuple[Optional[AISqlResponse], Optional[str]]:
    """
    Sends the user prompt and database schema to the configured DeepSeek HTTP API
    (or compatible endpoint), forcing a structured JSON response with fields
    'explanation' and 'sqlQuery'.
    """
    try:
        # Select the best available model
        model_name = select_best_model()

        # The prompt instructs the AI on its role, the schema, and the desired output.
        prompt = (
            "You are an expert Text-to-SQL agent. Your task is to analyze the user's "
            "natural language query and the provided database schema, and then generate "
            "a single, valid, SELECT-only PostgreSQL query.\n\n"
            "You must also provide a brief, one-sentence natural language explanation "
            "of what the query is doing.\n\n"
            "Respond with a JSON object that contains exactly two fields: 'explanation' and 'sqlQuery'.\n\n"
            f"Database Schema:\n{schema}\n\n"
            f"User Query:\n\"{user_prompt}\""
        )

        # If OpenRouter API is configured, call it. Otherwise return an instructive error.
        api_url = os.getenv("DEEPSEEK_API_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        referer = os.getenv("OPENROUTER_REFERER", "https://github.com/streamlit-text-to-sql")
        title = os.getenv("OPENROUTER_TITLE", "Streamlit Text-to-SQL")
        
        if not api_url or not api_key:
            return None, (
                "OpenRouter API not configured. Set DEEPSEEK_API_URL (e.g., https://openrouter.ai/api/v1) "
                "and DEEPSEEK_API_KEY (must start with sk-or-) in your environment "
                "or .streamlit/secrets.toml to enable AI generation."
            )

        try:
            # Build the OpenRouter / OpenChat-style completions endpoint.
            # Accept either a base API like https://openrouter.ai/api or https://openrouter.ai/api/v1
            # or a full completions URL. This logic prevents doubling /chat/completions.
            base_api = api_url.rstrip('/')
            if base_api.endswith('/chat/completions'):
                endpoint = base_api
            else:
                if not base_api.endswith('/v1'):
                    base_api = base_api + '/v1'
                endpoint = f"{base_api}/chat/completions"

            # Required headers for OpenRouter API
            headers = {
                "Authorization": f"Bearer {api_key}",  # sk-or- key required
                "Content-Type": "application/json",
                "HTTP-Referer": referer or "https://github.com/streamlit-text-to-sql",  # Required by OpenRouter
                "X-Title": title or "Streamlit Text-to-SQL"  # Optional but helpful for OpenRouter stats
            }

            # Construct messages: system instruction + user message
            system_msg = (
                "You are an expert Text-to-SQL agent. Your task is to analyze the user's "
                "natural language query and the provided database schema, and then generate "
                "a single, valid, SELECT-only PostgreSQL query. You must also provide a brief, "
                "one-sentence natural language explanation of what the query is doing. "
                "Respond with a JSON object that contains exactly two fields: 'explanation' and 'sqlQuery'."
            )

            payload = {
                "model": model_name,
                "temperature": 0.7,  # Add some creativity but not too much
                "response_format": { "type": "json_object" },  # Request structured JSON
                "messages": [
                    {"role": "system", "content": system_msg + "\n\nDatabase Schema:\n" + schema},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2048  # Sufficient for JSON response
            }

            # Basic validation: OpenRouter keys start with 'sk-or-'
            if not api_key.startswith("sk-or-"):
                st.error("DEEPSEEK_API_KEY does not look like a valid OpenRouter key (must start with 'sk-or-').")
                return None, "Invalid DeepSeek API key; it must start with 'sk-or-'."

            # Debug: show exactly which endpoint will be called (helps catch missing /v1)
            try:
                st.write(f"Calling endpoint: {endpoint}")
            except Exception:
                # Streamlit might not be available in non-UI contexts; ignore errors here
                pass

            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.HTTPError as e:
                # Provide a clearer error message including status and body when available
                resp_obj = e.response
                status = resp_obj.status_code if resp_obj is not None else 'unknown'
                text = resp_obj.text if resp_obj is not None else str(e)
                try:
                    st.error(f"HTTP error: {status} - {text}")
                except Exception:
                    pass
                return None, f"DeepSeek/OpenRouter API returned HTTP {status}"
            except Exception as e:
                return None, f"Error calling DeepSeek/OpenRouter API: {str(e)}"

            # OpenRouter returns: { choices: [{ message: { content: "..." } }] }
            content = None
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices and isinstance(choices, list):
                    first = choices[0]
                    if isinstance(first, dict):
                        message = first.get("message", {})
                        if isinstance(message, dict):
                            content = message.get("content")

            if not content:
                return None, "OpenRouter API returned no message content in the expected format"

            # The model may return a JSON string. Try to parse it robustly.
            parsed = None
            if isinstance(content, dict):
                parsed = content
            else:
                # content is a string; try direct JSON parse
                try:
                    parsed = json.loads(content)
                except Exception:
                    # Attempt to extract the first JSON object inside the string (common when model wraps JSON in text)
                    start = content.find('{')
                    end = content.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        try:
                            parsed = json.loads(content[start:end+1])
                        except Exception:
                            parsed = None

            if isinstance(parsed, dict) and "explanation" in parsed and "sqlQuery" in parsed:
                return {"explanation": parsed["explanation"], "sqlQuery": parsed["sqlQuery"]}, None

            return None, "DeepSeek/OpenRouter API returned content but it did not contain the expected JSON fields"
        except Exception as e:
            return None, f"Error calling DeepSeek/OpenRouter API: {str(e)}"

    except Exception as e:
        # Handle API errors gracefully
        return None, f"Unexpected error generating AI response: {str(e)}"