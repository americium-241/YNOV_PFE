#!/usr/bin/env python3
import time
# Wait for Neo4j to start

from neo4j import GraphDatabase

print("Running Neo4j post-install script...")
time.sleep(5)
print("waited 5 seconds")
time.sleep(5)
print("waited 5 seconds")
time.sleep(5)
print("waited 5 seconds")
time.sleep(5)
print("waited 5 seconds")
from neo4j import GraphDatabase
import time

def wait_for_neo4j():
    driver = None
    for _ in range(10):  
        print(_)# try 10 times
        try:
            driver = GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "password"))
            # Optionally, run a simple query to ensure connection is functional
            with driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            break
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    if not driver:
        raise Exception("Could not connect to Neo4j after multiple attempts.")
    return driver

driver = wait_for_neo4j()
# Connection configurations

session = driver.session()

# Insert a simple graph: Two nodes and a relationship
insert_query = """
    CREATE (a:Person {name: "Alice"})-[:KNOWS]->(b:Person {name: "Bob"})
"""

session.run(insert_query)
print("Inserted a simple graph: Alice KNOWS Bob.")

session.close()
