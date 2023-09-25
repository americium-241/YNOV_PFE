import streamlit as st
import folium
from streamlit_folium import st_folium
from neo4j import GraphDatabase
from geopy.geocoders import Nominatim
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import logging
import os
os.environ["OPENAI_API_KEY"] ='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'
st.set_page_config(
    page_title="Votre éco chemin :D",  # Title for the page # Wide mode
)
# Initialize logging
logging.basicConfig(level=logging.INFO)


# Function to find the closest node to a given location
def find_closest_node(tx, location, label="GRID_ROUTE"):
    query = f"""
    MATCH (n:{label})
    WITH n, point.distance(point({{latitude: $lat, longitude: $long}}), n.geometry) AS dist
    ORDER BY dist ASC
    LIMIT 1
    RETURN n
    """
    result = tx.run(query, lat=location[0], long=location[1])
    return result.single().value()


def find_shortest_path(tx, start_node, end_node, modalities=["foot"], metric="time"):
    # Map modalities to corresponding link types
    link_type_map = {
        "foot": "GRID_ROUTE_LINK",
        "car": "GRID_ROUTE_CAR_LINK",
        "velo": "GRID_ROUTE_VELO_LINK"
    }
    
    # Convert selected modalities to link types
    link_types = [link_type_map.get(modality) for modality in modalities if modality in link_type_map]
    link_types_string = "|".join(link_types)  # Convert to a string format compatible with Cypher's regex
    
    # Use the chosen metric as the cost
    cost = "distance"  # Default to distance
    if metric == "time":
        cost = "time"
    elif metric == "carbon":
        cost = "carbon_rate"

    # Determine the graph projection to use based on the modalities
    graph_projection = "GRAPH_TIME"
    if set(modalities) == {"foot", "car"}:
        graph_projection = "GRAPH_TIME_CAR"
    elif set(modalities) == {"foot", "velo"}:
        graph_projection = "GRAPH_TIME_VCUB"
    print(start_node,end_node)
    # Modify the query to use the A* algorithm with the specified graph projection
    query = f"""
    MATCH (source), (target)
    WHERE id(source) = $start_id AND id(target) = $end_id
    CALL gds.shortestPath.astar.stream('{graph_projection}', {{
        sourceNode: source,
        targetNode: target,
        latitudeProperty: 'latitude',
        longitudeProperty: 'longitude',
        relationshipWeightProperty: '{cost}'
    }})
    YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
    RETURN nodeIds
    """
    result = tx.run(query, start_id=start_node.id, end_id=end_node.id)

    # Extracting the node IDs from the result
    node_ids = [record["nodeIds"] for record in result][0]  # Assuming you only have one result

    return node_ids

def fetch_node_details(tx, node_ids):
    query = """
    UNWIND $node_ids AS node_id
    MATCH (n) WHERE id(n) = node_id
    RETURN id(n) AS NodeID, n.road_id AS RoadID, n.latitude AS Latitude, n.longitude AS Longitude, labels(n) AS Labels
    """
    result = tx.run(query, node_ids=node_ids)
    return [record for record in result]





modality_colors = {
    "GRID_ROUTE": "blue",
    "GRID_ROUTE_CAR": "red",
    "GRID_ROUTE_VELO": "green",
    "PARKING" : "yellow",
    "VCUB" : "black"
}

# Initialize the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

#center title
st.header("Votre éco chemin", anchor='center')
# Info Bubble

path_nodes=None



st.markdown("---")
st.info("""
**This app** finds the shortest path between two locations in France.
The app uses the [OpenStreetMap](https://www.openstreetmap.org/) data and the [Neo4j](https://neo4j.com/) graph database.
The app is built using [Streamlit](https://www.streamlit.io/).
""")
# Get the textual address from the user
start_address = st.text_input("Enter starting address:", "gare")
start_address= start_address + ", Bordeaux, France"
end_address = st.text_input("Enter destination address:", "Saint-michel")
end_address= end_address + ", Bordeaux, France"
# Modality selection
# Modality selection
modalities = ["foot", "car", "velo"]
chosen_modalities = st.multiselect("Choose modalities:", modalities, default=["foot"])  # Default to ['foot']
# Default to 'foot'

# Metric selection
metrics = ["time", "carbon"]
chosen_metric = st.radio("Choose a metric:", metrics, index=0)  # Default to 'time'

# Button to trigger shortest path calculation
if st.button("Find Shortest Path"):
    logging.info("Finding Shortest Path...")# Initialize the geolocator

    start_location = geolocator.geocode(start_address)
    end_location = geolocator.geocode(end_address)

    # Check if the addresses are valid
    if start_location is None or end_location is None:
        st.error("Could not determine the locations based on the addresses provided.")
    else:
        start_location = [start_location.latitude, start_location.longitude]
        end_location = [end_location.latitude, end_location.longitude]
        logging.info(f"Start Location: {start_location}, End Location: {end_location}")

        # Connect to Neo4j
        driver = GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "password"))

        # Find closest nodes to the specified locations
        logging.info("Finding closest nodes...")
        with driver.session() as session:
            start_node = session.read_transaction(find_closest_node, start_location)
            end_node = session.read_transaction(find_closest_node, end_location)
            logging.info(f"Start Node ID: {start_node.id}, End Node ID: {end_node.id}")
            
            # Fetch the shortest path between the specified nodes
            logging.info("Calculating shortest path...")
            path_nodes = session.read_transaction(find_shortest_path, start_node, end_node, chosen_modalities, chosen_metric)
            logging.info(path_nodes)
            node_details = session.read_transaction(fetch_node_details, path_nodes)
            logging.info(node_details)
        
        # Close the connection
        driver.close()
        
    # Check if there is a path
    if not path_nodes:
        st.error("No path found between the specified locations.")
    else:
        # Folium map
        m = folium.Map(location=start_location, zoom_start=10)

        # Extract locations for the path line
        locations = []
        unique_road_ids = set()
        previous_modality = None
        modality_changes = 0

            # Extract locations for the path line
        locations = []
        unique_road_ids = set()
        for i in range(1, len(node_details)):
            start = [node_details[i-1]['Latitude'], node_details[i-1]['Longitude']]
            end = [node_details[i]['Latitude'], node_details[i]['Longitude']]
            
            modality = next((label for label in node_details[i]['Labels'] if label in modality_colors), None)
            color = modality_colors.get(modality, "black")  # default to black if modality is not recognized

            folium.PolyLine(locations=[start, end], color=color, weight=2.5, opacity=1).add_to(m)

        
        # Adding a line between the markers to visualize the path
        #folium.PolyLine(locations=locations, color="blue", weight=2.5, opacity=1).add_to(m)

        # Adding markers for the start and end points
        folium.Marker(start_location, popup="Start", icon=folium.Icon(color='green')).add_to(m)
        folium.Marker(end_location, popup="End", icon=folium.Icon(color='red')).add_to(m)
        # Second Column: Map
        # Check if there is a path and display the map
        if 'm' in locals():  # Check if 'm' (the map object) has been created
            st_folium(m, width=800, height=600, returned_objects=[])
        logging.info("Shortest path visualization completed.")

        # Find the name of the road for each unique road_id using reverse geocoding
        road_names = []

from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import Optional, Dict, Any
from pydantic import BaseModel
class Map_tool(BaseTool):
    name = "map_display_tool"
    description = "Use this tool to display map to the user based on name of places, usage : 'start_place_name'|'destination_place_name'"

    def _run(
        self, start_end: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Use the tool."""
        start= start_end.split('|')[0]
        end = start_end.split('|')[1]
        start_location = geolocator.geocode(start_address)
        end_location = geolocator.geocode(end_address)

        # Check if the addresses are valid
        if start_location is None or end_location is None:
            st.error("Could not determine the locations based on the addresses provided.")
        else:
            start_location = [start_location.latitude, start_location.longitude]
            end_location = [end_location.latitude, end_location.longitude]
            logging.info(f"Start Location: {start_location}, End Location: {end_location}")

            # Connect to Neo4j
            driver = GraphDatabase.driver("neo4j://neo4j:7687", auth=("neo4j", "password"))

            # Find closest nodes to the specified locations
            logging.info("Finding closest nodes...")
            with driver.session() as session:
                start_node = session.read_transaction(find_closest_node, start_location)
                end_node = session.read_transaction(find_closest_node, end_location)
                logging.info(f"Start Node ID: {start_node.id}, End Node ID: {end_node.id}")
                
                # Fetch the shortest path between the specified nodes
                logging.info("Calculating shortest path...")
                path_nodes = session.read_transaction(find_shortest_path, start_node, end_node, chosen_modalities, chosen_metric)
                logging.info(path_nodes)
                node_details = session.read_transaction(fetch_node_details, path_nodes)
                logging.info(node_details)
            
            # Close the connection
            driver.close()
            
        # Check if there is a path
        if not path_nodes:
            st.error("No path found between the specified locations.")
        else:
            # Folium map
            m = folium.Map(location=start_location, zoom_start=10)

            # Extract locations for the path line
            locations = []
            unique_road_ids = set()
            previous_modality = None
            modality_changes = 0

                # Extract locations for the path line
            locations = []
            unique_road_ids = set()
            for i in range(1, len(node_details)):
                start = [node_details[i-1]['Latitude'], node_details[i-1]['Longitude']]
                end = [node_details[i]['Latitude'], node_details[i]['Longitude']]
                
                modality = next((label for label in node_details[i]['Labels'] if label in modality_colors), None)
                color = modality_colors.get(modality, "black")  # default to black if modality is not recognized

                folium.PolyLine(locations=[start, end], color=color, weight=2.5, opacity=1).add_to(m)

            
            # Adding a line between the markers to visualize the path
            #folium.PolyLine(locations=locations, color="blue", weight=2.5, opacity=1).add_to(m)

            # Adding markers for the start and end points
            folium.Marker(start_location, popup="Start", icon=folium.Icon(color='green')).add_to(m)
            folium.Marker(end_location, popup="End", icon=folium.Icon(color='red')).add_to(m)
            # Second Column: Map
            # Check if there is a path and display the map
            if 'm' in locals():  # Check if 'm' (the map object) has been created
                st_folium(m, width=800, height=600, returned_objects=[])
            logging.info("Shortest path visualization completed.")

            # Find the name of the road for each unique road_id using reverse geocoding
            road_names = []

        return 

    async def _arun(
        self, start_end: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """The navigation tool does not support async operations."""
        raise NotImplementedError("navigate does not support async")


llm = ChatOpenAI(temperature=0, streaming=True,model='gpt-4')
tools = load_tools(["ddg-search"])
tools.append(Map_tool())
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        info ='additional info:() utilisateur est toujours à Bordeaux en France, important : répond une liste avec un maximum de détail et de justification dans la final answer,langue:français.)'
        response = agent.run(info +prompt, callbacks=[st_callback])
        st.markdown(response)
# Global Footer
st.markdown("""
---
## Need Support? Found a Bug? 
* **Ask for Help:** [Start a Discussion](https://github.com/americium-241/YNOV_PFE/discussions)
* **Report a Bug or Request a Feature:** [Create an Issue](https://github.com/americium-241/YNOV_PFE/issues/new)
* **Browse Existing Issues:** [Check Issues and Pull Requests](https://github.com/americium-241/YNOV_PFE/issues)
""")
