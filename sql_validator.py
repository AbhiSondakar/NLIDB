import sqlparse
from typing import Tuple, Optional

# A set of keywords that are explicitly forbidden
FORBIDDEN_KEYWORDS = {
    'DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE', 'CREATE', 
    'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SET', 'EXECUTE', 'CALL', 'ATTACH',
    'DETACH', 'IMPORT', 'REINDEX', 'RELEASE', 'SAVEPOINT', 'VACUUM'
}

def validate_sql_query(sql_query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validates a SQL query to ensure it is safe to execute.
    1. Strips comments.
    2. Checks for a single SELECT-only statement.
    3. Blocks a list of forbidden keywords.
    
    Returns (validated_sql, None) on success, or (None, error_message) on failure.
    """
    try:
        # 1. Strip comments to prevent injection attacks (e.g., SELECT *; -- DROP TABLE)
        stripped_sql = sqlparse.format(sql_query, strip_comments=True).strip()
        if not stripped_sql:
            return None, "Validation Error: Query is empty after stripping comments."
        
        # 2. Parse the SQL and check for multiple statements
        parsed = sqlparse.parse(stripped_sql)
        if not parsed:
            return None, "Validation Error: Invalid SQL syntax."
        
        if len(parsed) > 1:
            return None, "Validation Error: Multiple SQL statements are not allowed."
        
        statement = parsed[0]
        
        # 3. Check if it's a SELECT statement
        stmt_type = statement.get_type()
        if stmt_type!= 'SELECT':
            return None, f"Validation Error: Only SELECT statements are allowed. Found: {stmt_type}"
        
        # 4. Final check for forbidden keywords (defense-in-depth)
        for token in statement.flatten():
            if token.is_keyword and token.value.upper() in FORBIDDEN_KEYWORDS:
                return None, f"Validation Error: Forbidden keyword '{token.value.upper()}' found."
                        
        return stripped_sql, None
        
    except Exception as e:
        return None, f"An unexpected validation error occurred: {str(e)}"