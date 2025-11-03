import os
import streamlit as st
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from models import Base  # Import the Base from models.py

# Define connection names. These MUST match the sections in.streamlit/secrets.toml
# Streamlit will automatically create secrets.toml from.env if it doesn't exist.
APP_DB_CONNECTION_NAME = "app_db"
DATA_DB_CONNECTION_NAME = "data_db"

@st.cache_resource
def get_app_db_connection():
    """
    Returns the Streamlit SQLConnection for the application's R/W database.
    Uses st.cache_resource to ensure the connection object is reused across reruns.
    """
    try:
        # First try to get URL from environment
        app_db_url = os.getenv("APP_DB_URL")
        
        if app_db_url:
            # If URL is in environment, use it directly
            conn = st.connection(APP_DB_CONNECTION_NAME, type="sql", url=app_db_url)
        else:
            # Otherwise, try to get from secrets.toml
            conn = st.connection(APP_DB_CONNECTION_NAME, type="sql")
        
        return conn
    except Exception as e:
        st.error(f"""
        Failed to connect to Application DB ({APP_DB_CONNECTION_NAME}): {e}
        
        Please ensure either:
        1. APP_DB_URL is set in your .env file, or
        2. url is set in [connections.app_db] in .streamlit/secrets.toml
        """)
        return None

@st.cache_resource
def get_data_db_connection():
    """
    Returns the Streamlit SQLConnection for the data warehouse (Read-Only).
    Uses st.cache_resource to ensure the connection object is reused.
    """
    try:
        # First try to get URL from environment
        data_db_url = os.getenv("DATA_DB_URL")
        
        if data_db_url:
            # If URL is in environment, use it directly
            conn = st.connection(DATA_DB_CONNECTION_NAME, type="sql", url=data_db_url)
        else:
            # Otherwise, try to get from secrets.toml
            conn = st.connection(DATA_DB_CONNECTION_NAME, type="sql")
        
        # Test the read-only connection
        with conn.session as s:
            s.execute(text("SELECT 1"))
        
        return conn
    except Exception as e:
        st.error(f"""
        Failed to connect to Data DB ({DATA_DB_CONNECTION_NAME}): {e}
        
        Please ensure either:
        1. DATA_DB_URL is set in your .env file, or
        2. url is set in [connections.data_db] in .streamlit/secrets.toml
        
        The read-only user must have CONNECT and SELECT privileges.
        """)
        return None

def init_db():
    """
    Initializes the application database by creating tables defined in models.py.
    This is a critical one-time setup step.
    """
    app_db = get_app_db_connection()
    if app_db:
        try:
            # Get the underlying SQLAlchemy engine from the Streamlit connection
            engine = app_db.engine
            # Create all tables defined in models.py that don't already exist
            Base.metadata.create_all(engine)
        except OperationalError as e:
            st.error(f"Error initializing application database: {e}. Is the database running?")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during DB initialization: {e}")
            st.stop()