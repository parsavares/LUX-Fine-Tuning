import duckdb

try:
    conn = duckdb.connect('data/test.duckdb')
    audio_data = conn.execute("SELECT audio FROM data LIMIT 1").fetchall()
    print(f"Audio data: {audio_data}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
