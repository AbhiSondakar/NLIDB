import streamlit as st
import pandas as pd
from uuid import UUID

# Local imports
import database
from models import ChatSession, ChatMessage
import ai_service
import sql_validator
import sql_executor

# --- APPLICATION SETUP ---

st.set_page_config(page_title="Text-to-SQL Agent", layout="wide")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize database (create tables if they don't exist)
database.init_db()

# Get database connections
# We must use the.engine attribute for services, 
# and the.session context manager for ORM operations.
app_db_conn = database.get_app_db_connection()
data_db_conn = database.get_data_db_connection()

if not app_db_conn or not data_db_conn:
    st.error("Failed to initialize database connections. Please check your.env file and database status.")
    st.stop()

# --- SESSION STATE INITIALIZATION ---

# st.session_state is Streamlit's way of preserving state across reruns
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id: UUID | None = None

# --- CALLBACK FUNCTIONS (for st.button) ---

def set_active_chat(session_id: UUID):
    """Callback to set the active chat session."""
    st.session_state.active_chat_id = session_id

def new_chat_callback():
    """Callback to create a new chat session and set it as active."""
    try:
        with app_db_conn.session as s:
            # Get count of sessions to create a default title
            session_count = s.query(ChatSession).count()
            
            new_session = ChatSession(title=f"Chat {session_count + 1}")
            s.add(new_session)
            s.commit()
            st.session_state.active_chat_id = new_session.id
            
    except Exception as e:
        st.error(f"Error creating new chat: {e}")

# --- HELPER FUNCTIONS ---

def save_message(session_id: UUID, role: str, content: dict):
    """Saves a message to the application database."""
    try:
        with app_db_conn.session as s:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content
            )
            s.add(msg)
            s.commit()
    except Exception as e:
        st.error(f"Error saving message: {e}")

def display_chat_messages(session_id: UUID):
    """Queries and displays all messages for the active chat session."""
    try:
        with app_db_conn.session as s:
            messages = s.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at).all()
            
            for msg in messages:
                with st.chat_message(msg.role):
                    # Unpack the JSONB content based on role
                    if msg.role == "user":
                        st.markdown(msg.content.get("text", "No content"))
                    
                    elif msg.role == "assistant":
                        st.markdown(msg.content.get("explanation", "Here is the result:"))
                        st.code(msg.content.get("sqlQuery", "# No SQL generated"), language="sql")
                        
                        results_data = msg.content.get("results")
                        if results_data:
                            st.dataframe(pd.DataFrame(results_data))
                        else:
                            st.info("Query executed successfully but returned no data.")
                    
                    elif msg.role == "system":
                        # Display errors using st.error
                        st.error(msg.content.get("error", "An unknown system error occurred."))
    
    except Exception as e:
        st.error(f"Error loading chat history: {e}")

# --- SIDEBAR UI ---

with st.sidebar:
    st.title("Text-to-SQL Agent")
    st.button("New Chat", on_click=new_chat_callback, use_container_width=True)
    
    st.divider()
    st.markdown("### Chat History")
    
    try:
        # Load all chat sessions from the app database
        with app_db_conn.session as s:
            sessions = s.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        
        if not sessions:
            st.caption("No chat history yet.")
        
        for session in sessions:
            # Use a button for each session
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

# --- MAIN CHAT WINDOW UI ---

if st.session_state.active_chat_id is None:
    st.info("Select a chat or start a new one to begin.")

else:
    # Display all historical messages
    display_chat_messages(st.session_state.active_chat_id)
    
    # Get new user input
    if prompt := st.chat_input("Ask a question about your data..."):
        
        # Save and display user message
        save_message(st.session_state.active_chat_id, "user", {"text": prompt})
        
        # --- CORE AI/SQL PIPELINE ---
        with st.chat_message("assistant"):
            with st.spinner("Analyzing request and generating SQL..."):
                
                # 1. Get Schema
                schema = ai_service.get_database_schema(data_db_conn.engine)
                if "Error" in schema:
                    st.error(f"Failed to get schema: {schema}")
                    save_message(st.session_state.active_chat_id, "system", {"error": schema})
                    st.stop()
                
                # 2. Get AI Response (Explanation + SQL)
                ai_response, error = ai_service.get_ai_response(prompt, schema)
                if error:
                    st.error(error)
                    save_message(st.session_state.active_chat_id, "system", {"error": error})
                    st.stop()
                
                # 3. Validate SQL
                validated_sql, error = sql_validator.validate_sql_query(ai_response['sqlQuery'])
                if error:
                    st.error(error)
                    save_message(st.session_state.active_chat_id, "system", {"error": error})
                    st.stop()
                
                # 4. Execute SQL
                with st.spinner("Executing query..."):
                    results, error = sql_executor.execute_sql_query(validated_sql, data_db_conn.engine)
                    if error:
                        st.error(error)
                        save_message(st.session_state.active_chat_id, "system", {"error": error})
                        st.stop()
                
                # 5. Save and display success
                assistant_content = {
                    "explanation": ai_response['explanation'],
                    "sqlQuery": validated_sql,
                    "results": results
                }
                save_message(st.session_state.active_chat_id, "assistant", assistant_content)
                
                # Rerun to display the new messages in the correct order
                st.rerun()