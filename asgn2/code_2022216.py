# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                  # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)  # Count of trips for each stop
fare_rules = {}                     # Mapping of route IDs to fare information
merged_fare_df = None               # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create knowledge base by populating global variables with information from loaded datasets.
    It establishes the relationships between routes, trips, stops, and fare rules.

    Returns:
        None
    """
    global route_to_stops, trip_to_route, stop_trip_count, fare_rules, merged_fare_df
    #clear global dictionaries before creating knowledge base
    route_to_stops.clear()
    trip_to_route.clear()
    stop_trip_count.clear()
    fare_rules.clear()

    # Create trip_id to route_id mapping
    for it in df_trips.index:
        trip_to_route[df_trips['trip_id'][it]] = df_trips['route_id'][it]

    # Map route_id to a list of stops in order of their sequence
    for it in df_stop_times.index:
        route_to_stops[trip_to_route[df_stop_times['trip_id'][it]]].append(df_stop_times['stop_id'][it])

    # Ensure each route only has unique stops
    s = set()
    for rt in route_to_stops:
        l = []
        for st in route_to_stops[rt]:
            if st not in s:
                s.add(st)
                l.append(st)
        route_to_stops[rt].clear()
        route_to_stops[rt] = l
        s.clear()

    # Count trips per stop
    for it in df_stop_times.index:
        stop_trip_count[df_stop_times['stop_id'][it]] += 1

    # Create fare rules for routes
    fare_rules = defaultdict(list)
    for it in df_fare_rules.index:
        fare_rules[df_fare_rules['route_id'][it]].append(df_fare_rules['fare_id'][it])

    # Merge fare rules and attributes into a single DataFrame
    merged_fare_df = df_fare_rules.merge(df_fare_attributes, on='fare_id')

# Function to find the top 5 busiest routes based on the number of trips
def get_busiest_routes():
    """
    Identify the top 5 busiest routes based on trip counts.

    Returns:
        list: A list of tuples, where each tuple contains:
              - route_id (int): The ID of the route.
              - trip_count (int): The number of trips for that route.
    """
    # pass  # Implementation here
    global trip_to_route
    route_trip_count = defaultdict(int)
    for it in trip_to_route.values():
        route_trip_count[it] += 1
    return sorted(route_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]

# Function to find the top 5 stops with the most frequent trips
def get_most_frequent_stops():
    """
    Identify the top 5 stops with the highest number of trips.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - trip_count (int): The number of trips for that stop.
    """
    # pass  # Implementation here
    global stop_trip_count
    return sorted(stop_trip_count.items(), key=lambda item: item[1], reverse=True)[:5]

# Function to find the top 5 busiest stops based on the number of routes passing through them
def get_top_5_busiest_stops():
    """
    Identify the top 5 stops with the highest number of different routes.

    Returns:
        list: A list of tuples, where each tuple contains:
              - stop_id (int): The ID of the stop.
              - route_count (int): The number of routes passing through that stop.
    """
    # pass  # Implementation here
    global route_to_stops
    stop_route_count = defaultdict(int)
    for it in route_to_stops.values():
        for i in it:
            stop_route_count[i] += 1
    return sorted(stop_route_count.items(), key=lambda item: item[1], reverse=True)[:5]

# Function to identify the top 5 pairs of stops with only one direct route between them
def get_stops_with_one_direct_route():
    """
    Identify the top 5 pairs of consecutive stops (start and end) connected by exactly one direct route. 
    The pairs are sorted by the combined frequency of trips passing through both stops.

    Returns:
        list: A list of tuples, where each tuple contains:
              - pair (tuple): A tuple with two stop IDs (stop_1, stop_2).
              - route_id (int): The ID of the route connecting the two stops.
    """
    # pass  # Implementation here
    global route_to_stops, stop_trip_count
    stop_pairs = defaultdict(list)
    for route, stops in route_to_stops.items():
        for it in range(len(stops)-1):
            pair = (stops[it], stops[it+1])
            stop_pairs[pair].append(route)

    stop_pairs_1_route = [(pair, route[0]) for pair, route in stop_pairs.items() if len(route) == 1]
    stop_pairs_1_route_dict = {}
    for it in stop_pairs_1_route:
        stop_pairs_1_route_dict[it] = stop_trip_count[it[0][0]] + stop_trip_count[it[0][1]]
    return sorted(stop_pairs_1_route_dict.keys(), key=lambda item: stop_pairs_1_route_dict[item], reverse=True)[:5]

# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
def visualize_stop_route_graph_interactive(route_to_stops):
    """
    Visualize the stop-route graph using Plotly for interactive exploration.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # pass  # Implementation here
    G = nx.DiGraph()
    for route, stops in route_to_stops.items():
        for i in range(len(stops) - 1):
            G.add_edge(stops[i], stops[i + 1], route=route) #add graph edges
    pos = nx.spring_layout(G, seed=42)
    #create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines'
    )
    #create node trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', text=node_text, textposition="top center", hoverinfo='text',
        marker=dict( showscale=True, colorscale='Viridis', size=10, line_width=2,
                    colorbar=dict( thickness=15, title='Node Connections', xanchor='left', titleside='right'))
    )
    #set node colors based on connections
    node_adj = []
    node_colors = []
    for node, adj in G.adjacency():
        node_adj.append(len(adj))
        node_colors.append(len(adj))
    node_trace.marker.color = node_colors
    node_trace.marker.size = [10 + adj * 2 for adj in node_adj]
    #create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Knowledge Base Graph', titlefont_size=16,
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all valid routes between two stops using a brute-force method.

    Args:
        start_stop (int): The ID of the starting stop.
        end_stop (int): The ID of the ending stop.

    Returns:
        list: A list of route IDs (int) that connect the two stops directly.
    """
    # pass  # Implementation here
    global route_to_stops
    direct_routes = []
    for route, stops in route_to_stops.items():
        #we dont check ordering here because we consider routes to be bidirectional
        if start_stop in stops and end_stop in stops:
            direct_routes.append(route)
    return direct_routes

# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, BoardRoute, TransferRoute, PDDL, X, Y, Z, R, R1, R2')
def initialize_datalog():
    """
    Initialize Datalog terms and predicates for reasoning about routes and stops.

    Returns:
        None
    """
    pyDatalog.clear()  # Clear previous terms
    print("Terms initialized: DirectRoute, RouteHasStop, OptimalRoute, BoardRoute, TransferRoute, PDDL")  # Confirmation print

    # Define Datalog predicates
    @pyDatalog.program()
    def _():
        DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y)
        OptimalRoute(R1, R2, X, Y, Z) <= DirectRoute(R1, X, Z) & DirectRoute(R2, Z, Y) & (R1!=R2)
        BoardRoute(R, X) <= RouteHasStop(R, X)
        TransferRoute(R1, R2, Z) <= RouteHasStop(R1, Z) & RouteHasStop(R2, Z) & (R1!=R2)
        PDDL(R1, R2, X, Y, Z) <= BoardRoute(R1, X) & TransferRoute(R1, R2, Z) & BoardRoute(R2, Y)

    global route_to_stops
    create_kb()  # Populate the knowledge base
    add_route_data(route_to_stops)  # Add route data to Datalog

# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add the route data to Datalog for reasoning.

    Args:
        route_to_stops (dict): A dictionary mapping route IDs to lists of stops.

    Returns:
        None
    """
    # pass  # Implementation here
    for route, stops in route_to_stops.items():
        for stop in stops:
            + RouteHasStop(route, stop)

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query for direct routes between two stops.

    Args:
        start (int): The ID of the starting stop.
        end (int): The ID of the ending stop.

    Returns:
        list: A sorted list of route IDs (str) connecting the two stops.
    """
    # pass  # Implementation here
    result = DirectRoute(R, start, end)
    return [route[0] for route in result]

# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform forward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # pass  # Implementation here
    # -------- procedural method --------
    # step1 = DirectRoute(R1, start_stop_id, stop_id_to_include)
    # step2 = DirectRoute(R2, stop_id_to_include, end_stop_id)
    # if len(step1)==0 or len(step2)==0:
    #     return []
    # optimal_routes = []
    # for r1 in step1:
    #     for r2 in step2:
    #         if r1 != r2:
    #             optimal_routes.append((r1[0], stop_id_to_include, r2[0]))
    # return optimal_routes
    # -------- declarative method --------
    result = OptimalRoute(R1, R2, start_stop_id, end_stop_id, stop_id_to_include)
    return [(route1, stop_id_to_include, route2) for (route1, route2) in result]

# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Perform backward chaining to find optimal routes considering transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID where a transfer occurs.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # pass  # Implementation here
    # -------- procedural method --------
    # step1 = DirectRoute(R1, end_stop_id, stop_id_to_include)
    # step2 = DirectRoute(R2, stop_id_to_include, start_stop_id)
    # if len(step1)==0 or len(step2)==0:
    #     return []
    # optimal_routes = []
    # for r1 in step1:
    #     for r2 in step2:
    #         if r1 != r2:
    #             optimal_routes.append((r1[0], stop_id_to_include, r2[0]))
    # return optimal_routes
    # -------- declarative method --------
    result = OptimalRoute(R1, R2, end_stop_id, start_stop_id, stop_id_to_include)
    return [(route1, stop_id_to_include, route2) for (route1, route2) in result]

# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implement PDDL-style planning to find routes with optional transfers.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        stop_id_to_include (int): The stop ID for a transfer.
        max_transfers (int): The maximum number of transfers allowed.

    Returns:
        list: A list of unique paths (list of tuples) that satisfy the criteria, where each tuple contains:
              - route_id (int): The ID of the route.
              - stop_id (int): The ID of the stop.
    """
    # pass  # Implementation here
    #start stop is our initial state and end stop is our goal state
    result = PDDL(R1, R2, start_stop_id, end_stop_id, stop_id_to_include)
    return [(route1, stop_id_to_include, route2) for (route1, route2) in result]

# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here