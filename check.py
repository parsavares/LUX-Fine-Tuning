import duckdb

# Attempt to connect to the DuckDB database
try:
    conn = duckdb.connect('data/train.duckdb')
    print("Connection successful!")

    # Try fetching some data to confirm access
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables in the database: {tables}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
