import streamlit as st
import pandas as pd
from uuid import UUID
import time
import os

# Local imports
import database
from models import ChatSession, ChatMessage
import ai_service
import sql_validator
from test import sql_executor
from visualization_service import VisualizationService

# --- APPLICATION SETUP ---

st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize database (create tables if they don't exist)
database.init_db()

# Get database connections
app_db_conn = database.get_app_db_connection()
data_db_conn = database.get_data_db_connection()

if not app_db_conn or not data_db_conn:
    st.error("Failed to initialize database connections. Please check your .env file and database status.")
    st.stop()

# Validate API configuration on startup
model_name, api_key, api_url = ai_service.get_model_config()
if not all([model_name, api_key, api_url]):
    st.error("""
    ❌ No AI API configured!

    Please configure the following in your .env file:

    ```env
    OPENROUTER_API_KEY=sk-or-v1-your-api-key
    MODEL_NAME=deepseek/deepseek-chat
    OPENROUTER_API_URL=https://openrouter.ai/api/v1
    ```

    **Available Models:**
    - `deepseek/deepseek-chat` - Excellent for SQL (Recommended)
    - `google/gemini-2.0-flash-exp:free` - Fast with vision support
    - `anthropic/claude-3.5-sonnet` - Top-tier reasoning
    - `openai/gpt-4o` - Powerful general model

    Get your API key at: https://openrouter.ai/keys
    """)
    st.stop()

# Initialize visualization service
viz_service = VisualizationService()

# --- SESSION STATE INITIALIZATION ---

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id: UUID | None = None

if "request_count" not in st.session_state:
    st.session_state.request_count = 0

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# Rate limiting configuration
MAX_REQUESTS_PER_MINUTE = 10


# --- HELPER FUNCTIONS ---

def get_filtered_schema():
    """Get schema with optional table filtering."""
    table_whitelist = None
    whitelist_env = os.getenv("SCHEMA_TABLE_WHITELIST")
    if whitelist_env:
        table_whitelist = [t.strip() for t in whitelist_env.split(",")]

    return ai_service.get_database_schema(data_db_conn.engine, table_whitelist)


def check_rate_limit() -> bool:
    """Check if user has exceeded rate limit."""
    current_time = time.time()

    # Reset counter if a minute has passed
    if current_time - st.session_state.last_request_time > 60:
        st.session_state.request_count = 0
        st.session_state.last_request_time = current_time

    # Check limit
    if st.session_state.request_count >= MAX_REQUESTS_PER_MINUTE:
        return False

    st.session_state.request_count += 1
    return True


def save_message(session_id: UUID, role: str, content: dict) -> UUID:
    """Saves a message and returns the message ID for reliable updates."""
    try:
        with app_db_conn.session as s:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            s.add(msg)
            s.commit()
            return msg.id
    except Exception as e:
        st.error(f"Error saving message: {e}")
        raise


def display_chat_messages(session_id: UUID):
    """Queries and displays all messages for the active chat session."""
    try:
        with app_db_conn.session as s:
            messages = s.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()

            for msg in messages:
                with st.chat_message(msg.role):
                    if msg.role == "user":
                        st.markdown(msg.content.get("text", "No content"))

                    elif msg.role == "assistant":
                        st.markdown(msg.content.get("explanation", "Here is the result:"))

                        sql_query = msg.content.get("sqlQuery", "# No SQL generated")
                        st.code(sql_query, language="sql")

                        # Show query stats if available
                        stats = sql_validator.get_validation_stats(sql_query)
                        if stats:
                            with st.expander("📊 Query Statistics"):
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Joins", stats.get('num_joins', 0))
                                col2.metric("Subqueries", stats.get('num_subqueries', 0))
                                col3.metric("Has Limit", "✅" if stats.get('has_limit') else "❌")

                        results_data = msg.content.get("results")
                        if results_data:
                            df = pd.DataFrame(results_data)
                            st.dataframe(df, use_container_width=True)
                            st.caption(f"Returned {len(df)} rows")

                            # Display saved visualization if exists
                            viz_data = msg.content.get("visualization")
                            if viz_data:
                                with st.expander("📈 Saved Visualization", expanded=False):
                                    st.caption(f"Chart Type: {viz_data['chart_type'].title()}")
                                    st.image(f"data:image/png;base64,{viz_data['image']}")
                        else:
                            st.info("Query executed successfully but returned no data.")

                    elif msg.role == "system":
                        st.error(msg.content.get("error", "An unknown system error occurred."))

    except Exception as e:
        st.error(f"Error loading chat history: {e}")


# --- CALLBACK FUNCTIONS ---

def set_active_chat(session_id: UUID):
    """Callback to set the active chat session."""
    st.session_state.active_chat_id = session_id


def new_chat_callback():
    """Callback to create a new chat session and set it as active."""
    try:
        with app_db_conn.session as s:
            session_count = s.query(ChatSession).count()
            new_session = ChatSession(title=f"Chat {session_count + 1}")
            s.add(new_session)
            s.commit()
            new_session_id = new_session.id

        st.session_state.active_chat_id = new_session_id

    except Exception as e:
        st.error(f"Error creating new chat: {e}")


def clear_schema_cache():
    """Callback to clear the schema cache."""
    ai_service.get_database_schema.clear()
    st.success("Schema cache cleared!")


# --- SIDEBAR UI ---

with st.sidebar:
    st.title("📊 Text-to-SQL Agent")

    # Display current model configuration
    if model_name:
        st.success(f"✅ Connected to OpenRouter")
        st.info(f"🤖 Model: **{model_name}**")

        # Model customization tip
        with st.expander("ℹ️ Change Model"):
            st.markdown("""
            To use a different model, update `MODEL_NAME` in your `.env` file:

            **Popular Options:**
            - `deepseek/deepseek-chat` (SQL expert)
            - `google/gemini-2.0-flash-exp:free` (Fast & free)
            - `anthropic/claude-3.5-sonnet` (Best reasoning)
            - `openai/gpt-4o` (General purpose)

            Browse all models at: https://openrouter.ai/models
            """)

    col1, col2 = st.columns(2)
    with col1:
        st.button("New Chat", on_click=new_chat_callback, use_container_width=True)
    with col2:
        st.button("Clear Cache", on_click=clear_schema_cache, use_container_width=True)

    st.divider()
    st.markdown("### 💬 Chat History")

    try:
        with app_db_conn.session as s:
            sessions = s.query(ChatSession).order_by(ChatSession.created_at.desc()).all()

        if not sessions:
            st.caption("No chat history yet.")

        for session in sessions:
            st.button(
                session.title,
                key=f"session_btn_{session.id}",
                on_click=set_active_chat,
                args=(session.id,),
                use_container_width=True,
                type="primary" if st.session_state.active_chat_id == session.id else "secondary"
            )
    except Exception as e:
        st.error(f"Could not load chat sessions: {e}")

    # Footer with stats
    st.divider()
    st.caption(f"⚡ Rate Limit: {st.session_state.request_count}/{MAX_REQUESTS_PER_MINUTE} requests/min")
    st.caption("📈 Visualization: Enabled")

# --- MAIN CHAT WINDOW UI ---

if st.session_state.active_chat_id is None:
    st.info("👈 Select a chat from the sidebar or start a new one to begin.")

    st.markdown("""
    ### Welcome to Text-to-SQL Agent! 🚀

    This app converts your natural language questions into SQL queries and executes them against your database.

    **Example questions:**
    - "Show me all tables in the database"
    - "How many users do we have?"
    - "What are the top 10 products by sales?"
    - "Show me users who signed up last month"

    **Features:**
    - ✅ Automatic SQL generation
    - ✅ Query validation and safety checks
    - ✅ Chat history persistence
    - ✅ Customizable AI models via .env
    - ✅ Intelligent data visualization

    **Current Model:** `{model_name}`
    """)

else:
    # Display all historical messages
    display_chat_messages(st.session_state.active_chat_id)

    # Get new user input
    if prompt := st.chat_input("Ask a question about your data..."):

        # Input validation
        if len(prompt) > 1000:
            st.error("❌ Query too long. Please keep it under 1000 characters.")
            st.stop()

        if not prompt.strip():
            st.error("❌ Please enter a valid question.")
            st.stop()

        # Rate limiting
        if not check_rate_limit():
            st.error(
                f"❌ Rate limit exceeded. Please wait before sending more requests. ({MAX_REQUESTS_PER_MINUTE} requests per minute)")
            st.stop()

        # Save and display user message
        try:
            save_message(st.session_state.active_chat_id, "user", {"text": prompt})
        except Exception as e:
            st.error(f"Failed to save message: {e}")
            st.stop()

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- CORE AI/SQL PIPELINE ---
        with st.chat_message("assistant"):
            try:
                with st.spinner("Analyzing request and generating SQL..."):

                    # 1. Get Schema with filtering
                    schema = get_filtered_schema()
                    if "Error" in schema:
                        error_msg = f"Failed to get schema: {schema}"
                        st.error(error_msg)
                        save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                        st.stop()

                    # 2. Get AI Response (Explanation + SQL)
                    ai_response, error = ai_service.get_ai_response(prompt, schema)
                    if error:
                        st.error(f"AI Generation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    # 3. Validate SQL
                    validated_sql, error = sql_validator.validate_sql_query(ai_response['sqlQuery'])
                    if error:
                        st.error(f"SQL Validation Error: {error}")
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()

                    # 4. Execute SQL
                    with st.spinner("Executing query..."):
                        results, error = sql_executor.execute_sql_query(validated_sql, data_db_conn.engine)
                        if error:
                            st.error(f"Execution Error: {error}")
                            save_message(st.session_state.active_chat_id, "system", {"error": error})
                            st.stop()

                    # 5. Prepare assistant content
                    assistant_content = {
                        "explanation": ai_response['explanation'],
                        "sqlQuery": validated_sql,
                        "results": results
                    }

                    # 6. Generate visualization if data exists
                    if results:
                        df = pd.DataFrame(results)
                        viz_analysis = viz_service.analyze_data_for_viz(df)

                        if viz_analysis['can_visualize']:
                            try:
                                chart_image = viz_service.create_chart(
                                    df,
                                    viz_analysis['recommended_chart'],
                                    x=viz_analysis.get('x_column'),
                                    y=viz_analysis.get('y_column'),
                                    title=f"{prompt[:50]}... - {viz_analysis['recommended_chart'].title()} Chart"
                                )
                                assistant_content['visualization'] = {
                                    'chart_type': viz_analysis['recommended_chart'],
                                    'image': chart_image
                                }
                            except Exception as viz_error:
                                st.warning(f"Visualization skipped: {str(viz_error)}")

                    # 7. Display results
                    st.markdown(assistant_content["explanation"])
                    st.code(validated_sql, language="sql")

                    # Show query stats
                    stats = sql_validator.get_validation_stats(validated_sql)
                    if stats:
                        with st.expander("📊 Query Statistics"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Joins", stats.get('num_joins', 0))
                            col2.metric("Subqueries", stats.get('num_subqueries', 0))
                            col3.metric("Has Limit", "✅" if stats.get('has_limit') else "❌")

                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        st.caption(f"Returned {len(df)} rows")

                        # --- VISUALIZATION SECTION ---
                        if 'visualization' in assistant_content:
                            with st.expander("📈 Data Visualization", expanded=True):
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    chart_type = assistant_content['visualization']['chart_type']
                                    st.info(f"💡 Recommended: {chart_type.title()} Chart")
                                    st.caption(viz_analysis['reason'])

                                with col2:
                                    chart_options = ['auto', 'line', 'bar', 'pie', 'scatter', 'histogram', 'heatmap',
                                                     'boxplot']
                                    new_chart_type = st.selectbox(
                                        "Change Chart Type",
                                        chart_options,
                                        index=0,
                                        key=f"chart_select_{st.session_state.active_chat_id}"
                                    )

                                st.image(f"data:image/png;base64,{assistant_content['visualization']['image']}")

                                if new_chart_type != 'auto' and new_chart_type != chart_type:
                                    if st.button("🔄 Regenerate with Selected Type",
                                                 key=f"regen_{st.session_state.active_chat_id}"):
                                        try:
                                            with st.spinner("Creating new visualization..."):
                                                new_chart_image = viz_service.create_chart(
                                                    df,
                                                    new_chart_type,
                                                    x=viz_analysis.get('x_column'),
                                                    y=viz_analysis.get('y_column'),
                                                    title=f"{prompt[:50]}... - {new_chart_type.title()} Chart"
                                                )
                                                assistant_content['visualization'] = {
                                                    'chart_type': new_chart_type,
                                                    'image': new_chart_image
                                                }
                                                st.rerun()
                                        except Exception as viz_error:
                                            st.error(f"Visualization Error: {str(viz_error)}")
                        else:
                            st.info(f"ℹ️ Visualization not available: {viz_analysis['reason']}")
                    else:
                        st.info("Query executed successfully but returned no data.")

                    # 8. Save complete message to database
                    save_message(st.session_state.active_chat_id, "assistant", assistant_content)

            except Exception as e:
                error_msg = f"Unexpected error in pipeline: {str(e)}"
                st.error(error_msg)
                try:
                    save_message(st.session_state.active_chat_id, "system", {"error": error_msg})
                except:
                    pass
