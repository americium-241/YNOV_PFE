#!/usr/bin/env python3

from neo4j import GraphDatabase
import time

def wait_for_neo4j():
    driver = None
    for _ in range(15):  
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


import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from neo4j import GraphDatabase
import requests
import py7zr

def read_and_filter_shapefile(file_path, bbox):
    """Load and filter a shapefile based on a bounding box."""
    minx, miny, maxx, maxy = bbox
    bbox_geom = box(minx, miny, maxx, maxy)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_geom]}, crs="epsg:4326")
    input_crs = gpd.read_file(file_path, rows=1).crs
    bbox_gdf = bbox_gdf.to_crs(input_crs)
    transformed_bbox = bbox_gdf['geometry'][0]
    df = gpd.read_file(file_path, bbox=transformed_bbox)
    return df.to_crs(epsg=4326)


def extract_coordinates(df):
    """Extract coordinates from GeoDataFrame and return as a DataFrame."""
    list_of_temp_dfs = []
    for i, row in df.iterrows():
        line = row['geometry']
        coords = list(line.coords)
        temp_df = pd.DataFrame(coords, columns=['longitude', 'latitude'])
        temp_df['id'] = i
        list_of_temp_dfs.append(temp_df)
    
    coords_df = pd.concat(list_of_temp_dfs, ignore_index=True)
    return coords_df

def add_nodes(tx, nodes):
    """Add nodes to the Neo4j database."""
    query = """
        UNWIND $nodes AS node
        CREATE (n:GRID_ROUTE {
            latitude: node.latitude, 
            longitude: node.longitude,
            road_id: node.road_id,
            id: node.id,
            geometry: point({longitude: node.longitude, latitude: node.latitude})
        })
    """
    tx.run(query, nodes=nodes)
    
def create_all_links(tx):
    """Create all links between the nodes in the Neo4j database."""
    q = """
    MATCH (node1:GRID_ROUTE), (node2:GRID_ROUTE)
    WHERE node1 <> node2 AND node1.road_id = node2.road_id
    WITH node1, node2, point.distance(node1.geometry, node2.geometry) AS distance
    ORDER BY node1, distance
    WITH node1, collect({node: node2, distance: distance}) AS node2s
    UNWIND node2s[0..2] AS node2_data
    WITH node1, node2_data.node AS node2, node2_data.distance AS dist
    MERGE (node1)-[:GRID_ROUTE_LINK {distance: dist, time: dist / 50}]->(node2)
    """
    tx.run(q)
    
def build_database( bbox, batch_size):
    """Build the Neo4j database using the provided shapefile and bounding box."""

    url = "https://wxs.ign.fr/pfinqfa9win76fllnimpfmbi/telechargement/inspire/ROUTE500-France-2021$ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/file/ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    save_path = "ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    download_file(url, save_path)
    
    # Step 2: Extract the .7z file
    extract_path = "extracted_files/"
    extract_7z(save_path, extract_path)
    # Step 1: Load and filter the shapefile
    df = read_and_filter_shapefile("./"+extract_path+'ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/ROUTE500/1_DONNEES_LIVRAISON_2022-01-00175/R500_3-0_SHP_LAMB93_FXX-ED211/RESEAU_ROUTIER/TRONCON_ROUTE.shp', bbox)
    
    # Step 2: Extract the coordinates
    coords_df = extract_coordinates(df)
    
    # Step 3: Connect to the Neo4j database
    driver = GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "password"))
    
    # Step 4: Add nodes to the database in batches
    nodes = []
    for i in range(0, len(coords_df)):
        node = coords_df.iloc[i]
        nodes.append({
            'latitude': node['latitude'],
            'longitude': node['longitude'],
            'road_id': node['id'],
            'id': i
        })
        if (i + 1) % batch_size == 0:
            with driver.session() as session:
                session.write_transaction(add_nodes, nodes)
            nodes = []
    if nodes:
        with driver.session() as session:
            session.write_transaction(add_nodes, nodes)
    
    # Step 5: Create links between the nodes
    with driver.session() as session:
        session.write_transaction(create_all_links)
    
    return
    #PART TO OPTIMIZE
    # Step 6: Create links between nodes that are within a certain distance threshold
    # You can set this to whatever value you need
    q = "MATCH (n:GRID_ROUTE) RETURN id(n) AS id"

    # Define distance threshold
    threshold = 2

    with driver.session() as session:
        result = session.read_transaction(lambda tx: tx.run(q).data())

    # For each node, create bidirectional links to other nodes within the threshold
    for record in result:
        node_id = record['id']
        with driver.session() as session:
            session.write_transaction(create_links_for_node, node_id, threshold)
        
    # Step 7: Close the Neo4j driver



def create_links_for_node(tx, node_id, threshold):
    """
    For a given node (specified by its ID), this function creates bidirectional links
    to all other nodes that are within a certain distance threshold.
    """
    q = f"""
    MATCH (n1:GRID_ROUTE) WHERE id(n1) = {node_id}
    MATCH (n2:GRID_ROUTE)
    WHERE n1 <> n2
    AND point.distance(n1.geometry, n2.geometry) < {threshold}
    WITH n1, n2, point.distance(n1.geometry, n2.geometry) AS distance
    MERGE (n1)-[link1:GRID_ROUTE_LINK]->(n2)
    ON CREATE SET link1.distance = distance, link1.time = distance / 50
    MERGE (n2)-[link2:GRID_ROUTE_LINK]->(n1)
    ON CREATE SET link2.distance = distance, link2.time = distance / 50
    """
    tx.run(q)
    
def download_file(url, save_path):
    """Download a file from a URL and save it to a local path."""
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def extract_7z(file_path, extract_path):
    """Extract a .7z file to a specified directory."""
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=extract_path)
    

bbox = [-1.190665, 44.634885, -0.339224, 45.299462]
batch_size = 1000

build_database(bbox, batch_size)
driver.close()
    
print("Database build complete.")