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

# Function to fetch the shortest path between two nodes
def find_shortest_path(tx, start_node, end_node):
    query = """
    MATCH path = shortestPath((start)-[*]-(end))
    WHERE id(start) = $start_id AND id(end) = $end_id
    UNWIND nodes(path) AS node
    RETURN DISTINCT node.id AS NodeID, node.road_id AS RoadID, node.latitude AS Latitude, node.longitude AS Longitude
    """
    result = tx.run(query, start_id=start_node.id, end_id=end_node.id)
    return [record for record in result]

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
            path_nodes = session.read_transaction(find_shortest_path, start_node, end_node)
        
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
        for node in path_nodes:
            location = [node['Latitude'], node['Longitude']]
            locations.append(location)
            unique_road_ids.add(node['RoadID'])
        
        # Adding a line between the markers to visualize the path
        folium.PolyLine(locations=locations, color="blue", weight=2.5, opacity=1).add_to(m)

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
        with st.expander("liste étapes", expanded=True):
            for road_id in unique_road_ids:
                node = next((node for node in path_nodes if node['RoadID'] == road_id), None)
                if node:
                    location = (node['Latitude'], node['Longitude'])
                    address = geolocator.reverse(location)
                    print(address)
                    print(address.raw)
                    road_name = address.raw['address'].get('road', 'Unknown Road')
                    st.markdown("---")
                    st.info(f"**{road_name}**")
                    st.markdown("100m")
                    road_names.append(road_name)




llm = ChatOpenAI(temperature=0, streaming=True,model='gpt-4')
tools = load_tools(["ddg-search"])
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

