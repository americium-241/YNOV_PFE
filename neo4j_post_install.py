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
    MERGE (node2)-[:GRID_ROUTE_LINK {distance: dist, time: dist / 50}]->(node1)
    """
    tx.run(q)

def duplicate_nodes(tx, original_label, copy_label):
    """Duplicate nodes in the Neo4j database for the copied grid."""
    query = f"""
        MATCH (n:{original_label})
        CREATE (m:{copy_label} {{
            latitude: n.latitude, 
            longitude: n.longitude,
            road_id: n.road_id,
            intersection_id: n.intersection_id,
            id: n.id,
            geometry: point({{longitude: n.longitude, latitude: n.latitude}})
        }})
    """
    tx.run(query)

def duplicate_links(tx, original_label, original_rel_type, copy_label, copy_rel_type):
    """Duplicate relationships in the Neo4j database for the copied grid."""
    q = f"""
    MATCH (node1:{original_label})-[r:{original_rel_type}]->(node2:{original_label})
    MATCH (node1_copy:{copy_label} {{id: node1.id}})
    MATCH (node2_copy:{copy_label} {{id: node2.id}})
    WHERE NOT (node1_copy)-[:{copy_rel_type}]->(node2_copy)  // Avoid creating duplicates
    CREATE (node1_copy)-[:{copy_rel_type} {{distance: r.distance, time: r.time}}]->(node2_copy)
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
    MERGE (p)-[:GRID_ROUTE_LINK {distance: dist}]->(closest_grid)
    MERGE (closest_grid)-[:GRID_ROUTE_LINK {distance: dist}]->(p)
    """
    tx.run(query)

def link_parkings_to_closest_grid_car(tx):
    """Link parking nodes to the closest grid node."""
    query = """
    MATCH (p:PARKING), (g:GRID_ROUTE_CAR)
    WITH p, g, point.distance(p.geometry, g.geometry) AS distance
    ORDER BY p, distance
    WITH p, collect({grid: g, distance: distance}) AS grids
    UNWIND grids[0..1] AS closest_grid_data
    WITH p, closest_grid_data.grid AS closest_grid, closest_grid_data.distance AS dist
    MERGE (p)-[:GRID_ROUTE_CAR_LINK {distance: dist}]->(closest_grid)
    MERGE (closest_grid)-[:GRID_ROUTE_CAR_LINK {distance: dist}]->(p)
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
        session.write_transaction(link_parkings_to_closest_grid_car)
    print("Parking nodes added to the database.", flush=True)


def add_vcub_nodes(tx, vcubs):
    """Add VCUB nodes to the Neo4j database."""
    query = """
        UNWIND $vcubs AS vcub
        CREATE (v:VCUB {
            latitude: vcub.latitude, 
            longitude: vcub.longitude,
            insee: vcub.INSEE,
            commune: vcub.commune,
            gml_id: vcub.gml_id,
            GID: vcub.GID,
            IDENT: vcub.IDENT,
            TYPE: vcub.TYPE,
            NOM: vcub.NOM,
            ETAT: vcub.ETAT,
            NBPLACES: vcub.NBPLACES,
            NBVELOS: vcub.NBVELOS,
            NBELEC: vcub.NBELEC,
            NBCLASSIQ: vcub.NBCLASSIQ,
            CDATE: vcub.CDATE,
            MDATE: vcub.MDATE,
            code_commune: vcub.code_commune,
            GEOM_O: vcub.GEOM_O,
            geometry: point({longitude: vcub.longitude, latitude: vcub.latitude})
        })
    """
    tx.run(query, vcubs=vcubs)

def link_vcubs_to_closest_grid(tx):
    """Link VCUB nodes to the closest grid node."""
    query = """
    MATCH (v:VCUB), (g:GRID_ROUTE)
    WITH v, g, point.distance(v.geometry, g.geometry) AS distance
    ORDER BY v, distance
    WITH v, collect({grid: g, distance: distance}) AS grids
    UNWIND grids[0..1] AS closest_grid_data
    WITH v, closest_grid_data.grid AS closest_grid, closest_grid_data.distance AS dist
    MERGE (v)-[:GRID_ROUTE_LINK {distance: dist}]->(closest_grid)
    MERGE (closest_grid)-[:GRID_ROUTE_LINK {distance: dist}]->(v)
    """
    tx.run(query)

def link_vcubs_to_closest_grid_velo(tx):
    """Link VCUB nodes to the closest grid node."""
    query = """
    MATCH (v:VCUB), (g:GRID_ROUTE_VELO)
    WITH v, g, point.distance(v.geometry, g.geometry) AS distance
    ORDER BY v, distance
    WITH v, collect({grid: g, distance: distance}) AS grids
    UNWIND grids[0..1] AS closest_grid_data
    WITH v, closest_grid_data.grid AS closest_grid, closest_grid_data.distance AS dist
    MERGE (v)-[:GRID_ROUTE_VELO_LINK {distance: dist}]->(closest_grid)
    MERGE (closest_grid)-[:GRID_ROUTE_VELO_LINK {distance: dist}]->(v)
    """
    tx.run(query)


def load_and_insert_vcub_data():
    # Step 1: Load the CSV into a Pandas DataFrame
    vcub_df = pd.read_csv("/data/ci_vcub_p.csv", delimiter=";", usecols=["Geo Point", "INSEE", "commune", "gml_id", "GID", "IDENT", "TYPE", "NOM", "ETAT", "NBPLACES", "NBVELOS", "NBELEC", "NBCLASSIQ", "CDATE", "MDATE", "code_commune", "GEOM_O"])
    # Extract latitude and longitude from the "Geo Point" column
    vcub_df['latitude'] = vcub_df['Geo Point'].str.split(", ").str[0].astype(float)
    vcub_df['longitude'] = vcub_df['Geo Point'].str.split(", ").str[1].astype(float)
    vcub_df.drop(columns=["Geo Point"], inplace=True)
    
    # Convert the DataFrame to a list of dictionaries
    vcubs = vcub_df.to_dict(orient="records")
    
    # Step 2: Insert VCUB data into Neo4j
    with driver.session() as session:
        session.write_transaction(add_vcub_nodes, vcubs)
        session.write_transaction(link_vcubs_to_closest_grid)
        session.write_transaction(link_vcubs_to_closest_grid_velo)
    print("VCUB nodes added to the database.", flush=True)

# Call the function
def download_file(url, save_path):
    """Download a file from a URL and save it to a local path."""    
    # Headers with a custom User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'
    }

    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def add_time_property(tx):
    # For GRID_ROUTE_LINK (5 km/hr)
    grid_route_query = """
    MATCH ()-[r:GRID_ROUTE_LINK]->()
    SET r.time = r.distance / 1.4
    """
    tx.run(grid_route_query)

    # For GRID_ROUTE_CAR_LINK (50 km/hr)
    grid_route_car_query = """
    MATCH ()-[r:GRID_ROUTE_CAR_LINK]->()
    SET r.time = r.distance / 13.9
    """
    tx.run(grid_route_car_query)

    # For GRID_ROUTE_VELO_LINK (25 km/hr)
    grid_route_velo_query = """
    MATCH ()-[r:GRID_ROUTE_VELO_LINK]->()
    SET r.time = r.distance / 5.6
    """
    tx.run(grid_route_velo_query)

    return 

def add_carbon_property(tx):
    # Batch update for GRID_ROUTE_LINK
    grid_route_query = """
    CALL apoc.periodic.iterate(
        "MATCH ()-[r:GRID_ROUTE_LINK]->() RETURN r", 
        "SET r.carbon_rate = (r.distance * 0.000001) / r.time, r.carbon = (r.distance * 0.000001)", 
        {batchSize: 10000, iterateList: true, parallel: false}
    )
    """
    tx.run(grid_route_query)

    # Batch update for GRID_ROUTE_CAR_LINK
    grid_route_car_query = """
    CALL apoc.periodic.iterate(
        "MATCH ()-[r:GRID_ROUTE_CAR_LINK]->() RETURN r", 
        "SET r.carbon_rate = (r.distance * 0.0025) / r.time, r.carbon = (r.distance * 0.0025)",
        {batchSize: 10000, iterateList: true, parallel: false}
    )
    """
    tx.run(grid_route_car_query)

    # Batch update for GRID_ROUTE_VELO_LINK
    grid_route_velo_query = """
    CALL apoc.periodic.iterate(
        "MATCH ()-[r:GRID_ROUTE_VELO_LINK]->() RETURN r", 
        "SET r.carbon_rate = (r.distance * 0.000001) / r.time, r.carbon = (r.distance * 0.000001)",
        {batchSize: 10000, iterateList: true, parallel: false}
    )
    """
    tx.run(grid_route_velo_query)


def project_graph_time(tx):
    query = """
    CALL gds.graph.project(
        'GRAPH_TIME',
        {
            GRID_ROUTE_CAR: {
                label: 'GRID_ROUTE_CAR',
                properties: ['latitude', 'longitude']
            },
            GRID_ROUTE_VELO: {
                label: 'GRID_ROUTE_VELO',
                properties: ['latitude', 'longitude']
            },
            GRID_ROUTE: {
                label: 'GRID_ROUTE',
                properties: ['latitude', 'longitude']
            },
            PARKING: {
                label: 'PARKING',
                properties: ['latitude', 'longitude']
            },
            VCUB: {
                label: 'VCUB',
                properties: ['latitude', 'longitude']
            }
        },
        {
            GRID_ROUTE_CAR_LINK: {
                type: 'GRID_ROUTE_CAR_LINK',
                properties: ['time','carbon_rate']
            },
            GRID_ROUTE_LINK: {
                type: 'GRID_ROUTE_LINK',
                properties: ['time','carbon_rate']
            },
            GRID_ROUTE_VELO_LINK: {
                type: 'GRID_ROUTE_VELO_LINK',
                properties: ['time','carbon_rate']
            }
        }
    )
    """
    tx.run(query)
def project_graph_time_vcub(tx):
    query = """
    CALL gds.graph.project(
        'GRAPH_TIME_VCUB',
        {
            GRID_ROUTE_VELO: {
                label: 'GRID_ROUTE_VELO',
                properties: ['latitude', 'longitude']
            },
            GRID_ROUTE: {
                label: 'GRID_ROUTE',
                properties: ['latitude', 'longitude']
            },
            VCUB: {
                label: 'VCUB',
                properties: ['latitude', 'longitude']
            }
        },
        {
            GRID_ROUTE_LINK: {
                type: 'GRID_ROUTE_LINK',
                properties: ['time','carbon_rate']
            },
            GRID_ROUTE_VELO_LINK: {
                type: 'GRID_ROUTE_VELO_LINK',
                properties: ['time','carbon_rate']
            }
        }
    )
    """
    tx.run(query)

def project_graph_time_car(tx):
    query = """
    CALL gds.graph.project(
        'GRAPH_TIME_CAR',
        {
            GRID_ROUTE_CAR: {
                label: 'GRID_ROUTE_CAR',
                properties: ['latitude', 'longitude']
            },
            GRID_ROUTE: {
                label: 'GRID_ROUTE',
                properties: ['latitude', 'longitude']
            },
            PARKING: {
                label: 'PARKING',
                properties: ['latitude', 'longitude']
            }
        },
        {
            GRID_ROUTE_CAR_LINK: {
                type: 'GRID_ROUTE_CAR_LINK',
                properties: ['time','carbon_rate']
            },
            GRID_ROUTE_LINK: {
                type: 'GRID_ROUTE_LINK',
                properties: ['time','carbon_rate']
            }
        }
    )
    """
    tx.run(query)

def project_graph_time_car(tx):
    query = """
    CALL gds.graph.project(
        'GRAPH',
        {
            GRID_ROUTE: {
                label: 'GRID_ROUTE',
                properties: ['latitude', 'longitude']
            }
        },
        {
            GRID_ROUTE_LINK: {
                type: 'GRID_ROUTE_LINK',
                properties: ['time','carbon_rate']
            }
        }
    )
    """
    tx.run(query)
import os

def build_database(bbox, batch_size):
    """Build the Neo4j database using the provided shapefile and bounding box."""

    url = "https://wxs.ign.fr/pfinqfa9win76fllnimpfmbi/telechargement/inspire/ROUTE500-France-2021$ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/file/ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    save_path = "/data/ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03.7z"
    
    if not os.path.exists(save_path):
        try:
            download_file(url, save_path)
            print("Download complete.", flush=True)
        except Exception as e:
            print(f"Failed to download due to: {e}. Using local file.", flush=True)
    
    # Step 2: Extract the .7z file
    extract_path = "extracted_files/"
    extract_7z(save_path, extract_path)
    print("Extraction complete.", flush=True)

    # Step 3: Load and filter the shapefile
    shapefile_path = os.path.join(extract_path, 'ROUTE500_3-0__SHP_LAMB93_FXX_2021-11-03/ROUTE500/1_DONNEES_LIVRAISON_2022-01-00175/R500_3-0_SHP_LAMB93_FXX-ED211/RESEAU_ROUTIER/TRONCON_ROUTE.shp')
    df = read_and_filter_shapefile(shapefile_path, bbox)
    print("Shapefile loaded and filtered.", flush=True)

    # Step 4: Extract the coordinates
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
        # Duplicate nodes for car
        print("Duplicate nodes for car")
        session.write_transaction(duplicate_nodes, "GRID_ROUTE", "GRID_ROUTE_CAR")
        # Duplicate relationships for car
        print("Duplicate relationships for car")
        session.write_transaction(duplicate_links, "GRID_ROUTE", "GRID_ROUTE_LINK", "GRID_ROUTE_CAR", "GRID_ROUTE_CAR_LINK")
        
        # Duplicate nodes for velo
        print("Duplicate nodes for velo")
        session.write_transaction(duplicate_nodes, "GRID_ROUTE", "GRID_ROUTE_VELO")
        # Duplicate relationships for velo
        print("Duplicate relationships for velo")
        session.write_transaction(duplicate_links, "GRID_ROUTE", "GRID_ROUTE_LINK", "GRID_ROUTE_VELO", "GRID_ROUTE_VELO_LINK")
    print("Links created 1." , flush=True)  
    print("loading parking...")
    load_and_insert_parking_data()
    print("parking created and connected")
    print("loading velo...")
    load_and_insert_vcub_data()
    print("velo created and connected")
    with driver.session() as session:
        session.write_transaction(add_time_property)
        print("loading time")
        session.write_transaction(add_carbon_property)
        print("loading carbon")
        session.write_transaction(project_graph_time)       
        session.write_transaction(project_graph_time_car)       
        session.write_transaction(project_graph_time_vcub)
    print("carbon print and time set")

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
        MERGE (n2)-[:GRID_ROUTE_LINK {distance: 1, time: 1}]->(n1)
    """
    tx.run(query)

    


def extract_7z(file_path, extract_path):
    """Extract a .7z file to a specified directory."""
    with py7zr.SevenZipFile(file_path, mode='r') as archive:
        archive.extractall(path=extract_path)
    

bbox = [-1.190665, 44.634885, -0.339224, 45.299462]
batch_size = 1000

build_database(bbox, batch_size)
driver.close()
    
print("Database build complete.")