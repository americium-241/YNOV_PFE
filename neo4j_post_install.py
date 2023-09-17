#!/usr/bin/env python3

from neo4j import GraphDatabase
import time
import numpy as np
import pandas as pd

import numpy as np

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
def haversine(lat1, lon1, lat2, lon2):

    """

    Calculate the great circle distance between two points 

    on the earth (specified in decimal degrees)

    """

    # Convert decimal degrees to radians 

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a)) 

    # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.

    r = 6371

    return c * r * 1000
def assign_intersection_id_grid(df, threshold=3):
    """
    Highly optimized version to assign a unique intersection ID to points 
    that are within the specified distance threshold of each other, using a grid-based approach.
    
    Parameters:
    - df: DataFrame containing 'longitude', 'latitude', and 'id_road' columns.
    - threshold: Distance threshold in meters. Points closer than this threshold will be assigned the same intersection ID.
    
    Returns:
    - DataFrame with an additional 'intersection_id' column.
    """
    # Convert latitude and longitude to radians
    coords = np.radians(df[['latitude', 'longitude']].values)
    
    # Convert to Cartesian coordinates (meters) using Mercator projection
    R = 6378137  # Earth radius in meters
    x = R * coords[:, 1]
    y = R * np.log(np.tan(np.pi/4 + coords[:, 0]/2))
    
    # Create grid indices
    grid_size = threshold
    x_idx = (x / grid_size).astype(int)
    y_idx = (y / grid_size).astype(int)
    
    # Initialize intersection ID and visited set
    intersection_id = 0
    df['intersection_id'] = np.nan
    visited = set()
    
    # Iterate through points and assign intersection IDs
    for i in range(len(df)):
        if i not in visited:
            # Find points in the same or adjacent grid cells
            close_points = np.where(
                (x_idx >= x_idx[i] - 1) & (x_idx <= x_idx[i] + 1) &
                (y_idx >= y_idx[i] - 1) & (y_idx <= y_idx[i] + 1)
            )[0]
            
            # Check the actual distances
            close_points = close_points[
                haversine(coords[i, 0], coords[i, 1], coords[close_points, 0], coords[close_points, 1]) < threshold
            ]
            
            if len(close_points) > 1:
                # Assign a new intersection ID to these points
                df.loc[close_points, 'intersection_id'] = intersection_id
                visited.update(close_points)
                intersection_id += 1  # Increment the intersection ID for the next group
    
    return df

# Test the highly optimized grid-based function with the adjusted sample data



def add_nodes(tx, nodes):
    """Add nodes to the Neo4j database."""
    query = """
        UNWIND $nodes AS node
        CREATE (n:GRID_ROUTE {
            latitude: node.latitude, 
            longitude: node.longitude,
            road_id: node.road_id,
            intersection_id: node.intersection_id,
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


def add_parking_nodes(tx, parkings):
    """Add parking nodes to the Neo4j database."""
    query = """
        UNWIND $parkings AS parking
        CREATE (p:PARKING {
            latitude: parking.latitude, 
            longitude: parking.longitude,
            insee: parking.insee,
            ident: parking.ident,
            adresse: parking.adresse,
            nb_niv: parking.nb_niv,
            nom: parking.nom,
            total: parking.total,
            url: parking.url,
            geometry: point({longitude: parking.longitude, latitude: parking.latitude})
        })
    """
    tx.run(query, parkings=parkings)

def link_parkings_to_closest_grid(tx):
    """Link parking nodes to the closest grid node."""
    query = """
    MATCH (p:PARKING), (g:GRID_ROUTE)
    WITH p, g, point.distance(p.geometry, g.geometry) AS distance
    ORDER BY p, distance
    WITH p, collect({grid: g, distance: distance}) AS grids
    UNWIND grids[0..1] AS closest_grid_data
    WITH p, closest_grid_data.grid AS closest_grid, closest_grid_data.distance AS dist
    MERGE (p)-[:LINKS_TO {distance: dist}]->(closest_grid)
    """
    tx.run(query)


def load_and_insert_parking_data():
    # Step 1: Load the CSV into a Pandas DataFrame
    parking_df = pd.read_csv("/data/st_park_p.csv", delimiter=";", usecols=["Geo Point", "insee", "ident", "adresse", "nb_niv", "nom", "total", "url"])
    # Extract latitude and longitude from the "Geo Point" column
    parking_df['latitude'] = parking_df['Geo Point'].str.split(", ").str[0].astype(float)
    parking_df['longitude'] = parking_df['Geo Point'].str.split(", ").str[1].astype(float)
    parking_df.drop(columns=["Geo Point"], inplace=True)
    
    # Convert the DataFrame to a list of dictionaries
    parkings = parking_df.to_dict(orient="records")
    
    # Step 2: Insert parking data into Neo4j
    with driver.session() as session:
        session.write_transaction(add_parking_nodes, parkings)
        session.write_transaction(link_parkings_to_closest_grid)
    print("Parking nodes added to the database.", flush=True)

# Call the function


    
def build_database( bbox, batch_size):
    """Build the Neo4j database using the provided shapefile and bounding box."""

    url = "https://wxs.ign.fr/pfinqfa9win76fllnimpfmbi/telechargement/inspire/ROUTE500-France-2021$ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/file/ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    save_path = "ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    download_file(url, save_path)
    print("Download complete.", flush=True)
    
    # Step 2: Extract the .7z file
    extract_path = "extracted_files/"
    extract_7z(save_path, extract_path)
    print("Extraction complete.", flush=True)
    # Step 1: Load and filter the shapefile
    df = read_and_filter_shapefile("./"+extract_path+'ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/ROUTE500/1_DONNEES_LIVRAISON_2022-01-00175/R500_3-0_SHP_LAMB93_FXX-ED211/RESEAU_ROUTIER/TRONCON_ROUTE.shp', bbox)
    print("Shapefile loaded and filtered.", flush=True)
    # Step 2: Extract the coordinates
    coords_df = extract_coordinates(df)
    print("Coordinates extracted.", flush=True)
    coords_df = assign_intersection_id_grid(coords_df, threshold=3)
    print("Intersection IDs assigned.", flush=True)
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
            'intersection_id': node['intersection_id'],
            'id': i
        })
        if (i + 1) % batch_size == 0:
            with driver.session() as session:
                session.write_transaction(add_nodes, nodes)
            nodes = []
    if nodes:
        with driver.session() as session:
            session.write_transaction(add_nodes, nodes)
    print("Nodes added to the database.", flush=True)
    
    # Step 5: Create links between the nodes
    with driver.session() as session:
        session.write_transaction(create_all_links)
    print("Links created 0.", flush=True)
    with driver.session() as session:
        session.write_transaction(connect_intersection_nodes)
    print("Links created 1." , flush=True)  
    print("loading parking...")
    load_and_insert_parking_data()
    print("parking created and connected")
    return



def connect_intersection_nodes(tx):
    """Create relationships between nodes with the same intersection_id."""
    query = """
        MATCH (n:GRID_ROUTE)
        WITH n.intersection_id AS intersection, COLLECT(n) AS nodes
        WHERE SIZE(nodes) > 1
        UNWIND nodes AS n1
        WITH n1, nodes
        UNWIND nodes AS n2
        WITH n1, n2
        WHERE id(n1) < id(n2)
        MERGE (n1)-[:GRID_ROUTE_LINK {distance: 1, time: 1}]->(n2)
    """
    tx.run(query)

    
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