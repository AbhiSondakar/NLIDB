import os
import streamlit as st
import requests
import json
import re
from sqlalchemy import inspect, Engine
from typing import TypedDict, Tuple, Optional, Dict, Any, List


class AISqlResponse(TypedDict):
    explanation: str
    sqlQuery: str


def parse_json_response(content: str) -> Optional[dict]:
    """Robust JSON parsing with multiple fallback strategies."""
    if not content:
        return None

    if isinstance(content, dict):
        return content

    # Remove common text artifacts
    content = content.strip()

    # Remove markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()

    # Try direct JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract JSON from code blocks (alternative patterns)
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*"explanation"[^{}]*"sqlQuery"[^{}]*\}',
        r'\{.*?\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                cleaned = match.strip()
                cleaned = re.sub(r'```json|```', '', cleaned).strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Find first { and last } - last resort
    start = content.find('{')
    end = content.rfind('}')

    if start != -1 and end != -1 and end > start:
        try:
            json_str = content[start:end + 1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def get_model_config() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get model configuration from environment variables.
    Returns: (model_name, api_key, api_url)
    """
    model_name = os.getenv("MODEL_NAME", "deepseek/deepseek-chat")

    # Check OpenRouter configuration
    api_key = os.getenv("OPENROUTER_API_KEY")
    api_url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        return None, None, None

    return model_name, api_key, api_url


@st.cache_data(ttl=600)
def get_database_schema(_engine: Engine, table_whitelist: Optional[List[str]] = None) -> str:
    """Get database schema as CREATE TABLE statements with optional filtering."""
    try:
        inspector = inspect(_engine)
        schema_str = ""
        table_names = inspector.get_table_names()

        if not table_names:
            return "Error: No tables found in the database."

        if table_whitelist:
            table_names = [t for t in table_names if t in table_whitelist]
            if not table_names:
                return f"Error: No tables match whitelist: {table_whitelist}"

        max_tables = int(os.getenv("MAX_SCHEMA_TABLES", "50"))
        if len(table_names) > max_tables:
            st.warning(f"⚠️ Large schema detected: showing {max_tables} of {len(table_names)} tables.")
            table_names = table_names[:max_tables]

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

        return schema_str
    except Exception as e:
        return f"Error inspecting schema: {e}"


def call_openrouter_api(
        model: str,
        messages: list,
        api_key: str,
        api_url: str,
        image_url: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Call OpenRouter API with optional image support."""
    try:
        endpoint = api_url.rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            if not endpoint.endswith('/v1'):
                endpoint += '/v1'
            endpoint += '/chat/completions'

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/text-to-sql"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "Text-to-SQL Agent")
        }

        # Prepare messages with image if provided
        prepared_messages = []
        for msg in messages:
            if msg["role"] == "user" and image_url:
                prepared_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                prepared_messages.append(msg)

        payload = {
            "model": model,
            "temperature": 0.3,
            "messages": prepared_messages,
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        return content, None

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        return None, f"API error: {error_detail}"
    except Exception as e:
        return None, f"API error: {str(e)}"


def get_ai_response(
        user_prompt: str,
        schema: str,
        image_url: Optional[str] = None
) -> Tuple[Optional[AISqlResponse], Optional[str]]:
    """Send request to configured AI provider and get SQL response."""
    try:
        model_name, api_key, api_url = get_model_config()

        if not all([model_name, api_key, api_url]):
            return None, (
                "❌ No AI API configured!\n\n"
                "Please set the following in your .env file:\n"
                "- OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)\n"
                "- MODEL_NAME (e.g., deepseek/deepseek-chat)\n"
                "- OPENROUTER_API_URL (default: https://openrouter.ai/api/v1)"
            )

        # Display current model
        st.info(f"🤖 Using model: **{model_name}**")

        system_msg = (
            "You are an expert Text-to-SQL agent. Your ONLY job is to convert natural language questions into SQL queries.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST respond with ONLY a valid JSON object - nothing else\n"
            "2. The JSON must have exactly two fields: 'explanation' and 'sqlQuery'\n"
            "3. If the user asks for something impossible (like data not in the schema), still generate a reasonable query\n"
            "4. If a table/column doesn't exist, generate a query that shows what IS available\n"
            "5. NEVER say 'I cannot' or refuse - always provide a SQL query\n"
            "6. Do NOT include any markdown code blocks, just pure JSON\n\n"
            "Response format (copy this structure exactly):\n"
            '{"explanation": "Brief one-sentence description", "sqlQuery": "SELECT * FROM table_name LIMIT 10"}\n\n'
            f"Available Database Schema:\n{schema}\n\n"
            "Remember: Respond with ONLY the JSON object, no other text!"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ]

        content, error = call_openrouter_api(model_name, messages, api_key, api_url, image_url)

        if error:
            return None, error

        if not content:
            return None, "API returned empty response"

        parsed = parse_json_response(content)

        if not parsed:
            # If JSON parsing fails, try to extract useful information
            # and create a helpful response
            if "cannot" in content.lower() or "no table" in content.lower():
                # Model is refusing - provide a fallback query
                return {
                    "explanation": "Showing available tables in the database",
                    "sqlQuery": "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name LIMIT 20"
                }, None
            else:
                return None, f"Could not parse valid JSON from response. Model returned: {content[:300]}..."

        if "explanation" not in parsed or "sqlQuery" not in parsed:
            return None, f"Missing required fields. Got: {list(parsed.keys())}"

        return {
            "explanation": parsed["explanation"],
            "sqlQuery": parsed["sqlQuery"]
        }, None

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"
