import duckdb

# Path to your DuckDB file
train_file = '/home/users/pvares/paper/data/train.duckdb'

# Create a connection to DuckDB, directly connecting to the database file
conn = duckdb.connect(train_file)

# Check for tables
tables = conn.execute("SHOW TABLES").fetchall()

print("Tables in train.duckdb:", tables)

# If tables are present, query the first one
if tables:
    table_name = tables[0][0]
    print(f"Querying data from table: {table_name}")
    df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
    print(df.head())

# Close the connection when done
conn.close()
