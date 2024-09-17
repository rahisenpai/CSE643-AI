import numpy as np
import pickle
import heapq as pq
import time
import tracemalloc
import matplotlib.pyplot as plt

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def depth_limited_search(adj_matrix, curr_node, goal_node, depth, reached):
  #returns true if goal is found at depth, else false
  #base cases
  if curr_node == goal_node:
    return True #goal is found
  if depth == 0:
    return False #exhausted depth

  #find neighbors from adjacency matrix
  neighbors = []
  for i in range(len(adj_matrix)):
    if adj_matrix[curr_node][i] > 0: #condition for neighbors
      neighbors.append(i)

  #proceed with dfs
  for next_node in neighbors:
    #if next node is not already visited, avoid cycles
    if next_node not in reached:
      reached[next_node] = curr_node #add in reached
      #we are using process stack as the frontier
      if depth_limited_search(adj_matrix, next_node, goal_node, depth-1, reached):
        return True
  return False

def get_ids_path(adj_matrix, start_node, goal_node):
  #first we will perform ids
  flag = False #setting flag to check if we ever found goal node
  #checking at all depths
  for depth in range(len(adj_matrix)):
    #using a dictionary to store visited nodes (to avoid cycles) as key and parent nodes as value
    #this dictionary will be used to build the path from goal to start
    reached = {start_node:None}
    if depth_limited_search(adj_matrix, start_node, goal_node, depth, reached):
      flag = True #set flag true as we found goal node
      break #we found the goal and don't need to increase depth furhter

  if not flag: #goal node was never found at any depth
    return None

  #finding the path from goal to start
  path = [goal_node]
  node = goal_node
  while reached[node]: #loop will stop when it reaches start_node as its values is None
    path.append(reached[node])
    node = reached[node]
  path.reverse() #reverse the path from start to goal
  return path


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def bfs(dir, adj_matrix, frontier, reached, reached2):
  curr_node = frontier.pop(0) #pop the top node

  #find neighbors and their distance from adjacency matrix
  neighbors = []
  if dir == 'F': #forward direction, check outgoing edges
    for i in range(len(adj_matrix)):
      if adj_matrix[curr_node][i] > 0: #condition for neighbors
        neighbors.append(i)
  else: #direction is backward, check incoming edges
    for i in range(len(adj_matrix)):
      if adj_matrix[i][curr_node] > 0: #condition for neighbors
        neighbors.append(i)

  #proceed with bfs
  for next_node in neighbors:
    if next_node not in reached: #if child is not already visited
      reached[next_node] = curr_node #mark child as visited
      frontier.append(next_node) #add child to frontier
    if next_node in reached2: #check if this is an intersecting node
      return next_node
  return None

def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
  #first we will perform bidirectional search
  frontierF = [start_node] #queue for forward direction
  frontierB = [goal_node] #queue for backward direction
  reachedF = {start_node : None} #dictionary to store visited nodes and parents in forward direction
  reachedB = {goal_node : None} #dictionary to store visited nodes and parents in backward direction

  while frontierF and frontierB: #until any frontier is empty
    #forward direction bfs
    intersecting_node = bfs('F', adj_matrix, frontierF, reachedF, reachedB)
    if intersecting_node:
      break #node is not None, we found intersecting node
    #backward direction bfs
    intersecting_node = bfs('B', adj_matrix, frontierB, reachedB, reachedF)
    if intersecting_node:
      break #node is not None, we found intersecting node

  if not intersecting_node: #goal node was never found at any depth
    return None

  #finding the path
  path = [intersecting_node]
  node = intersecting_node
  while reachedF[node]: #path from intersecting_node to start_node
    path.append(reachedF[node])
    node = reachedF[node]
  path.reverse() #reverse to get path from start_node to intersecting_node
  node = intersecting_node
  while reachedB[node]: #path from intersecting_node to goal_node
    path.append(reachedB[node])
    node = reachedB[node]
  return path


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 9, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

def heuristic(node, start_node, goal_node, node_attributes):
  u = node_attributes[start_node]
  v = node_attributes[goal_node]
  w = node_attributes[node]
  distuw = np.sqrt(((u['x'] - w['x']) ** 2) + ((u['y'] - w['y']) ** 2))
  distwv = np.sqrt(((w['x'] - v['x']) ** 2) + ((w['y'] - v['y']) ** 2))
  return distuw + distwv

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
  #first we perform the astar search
  #priority queue
  frontier = [(0 + heuristic(start_node, start_node, goal_node, node_attributes), 0, start_node)]
  #dictionary to store visited nodes and distances and parents
  reached = {start_node: (0, None)}

  while frontier:
    temp, dist, curr_node = pq.heappop(frontier) #pop top node

    #we found the goal_node, reconstruct path
    if curr_node == goal_node:
      path = [goal_node]
      node = goal_node
      while reached[node][1]: #loop will stop when it reaches start_node as its values is None
        path.append(reached[node][1])
        node = reached[node][1]
      path.reverse() #reverse the path from start to goal
      return path

    #find neighbors from adjacency matrix
    neighbors = {}
    for i in range(len(adj_matrix)):
      if adj_matrix[curr_node][i] > 0: #condition for neighbors
        neighbors[i] = adj_matrix[curr_node][i]

    #check the neighbors
    for next_node in neighbors:
      new_dist = dist + neighbors[next_node]
      #if child is not visited or a shorter path is found
      if (next_node not in reached) or (new_dist < reached[next_node][0]):
        reached[next_node] = (new_dist, curr_node)
        pq.heappush(frontier, (new_dist + heuristic(next_node, start_node, goal_node, node_attributes), new_dist, next_node))
  return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

def proceed(dir, adj_matrix, frontier, reached, reached2, start_node, goal_node, node_attributes):
  temp, dist, curr_node = pq.heappop(frontier) #pop the top node

  #find neighbors and their distance from adjacency matrix
  neighbors = {}
  if dir == 'F': #forward direction, check outgoing edges
    for i in range(len(adj_matrix)):
      if adj_matrix[curr_node][i] > 0: #condition for neighbors
        neighbors[i] = adj_matrix[curr_node][i]
  else: #direction is backward, check incoming edges
    for i in range(len(adj_matrix)):
      if adj_matrix[i][curr_node] > 0: #condition for neighbors
        neighbors[i] = adj_matrix[i][curr_node]

  #check the neighbors
  for next_node in neighbors:
    new_dist = dist + neighbors[next_node]
    #if child is not visited or a shorter path is found
    if (next_node not in reached) or (new_dist < reached[next_node][0]):
      reached[next_node] = (new_dist, curr_node)
      pq.heappush(frontier, (new_dist + heuristic(next_node, start_node, goal_node, node_attributes), new_dist, next_node))
    if next_node in reached2: #we found intersecting node
      return next_node
  return None

def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
  #first we will perform bidirectional search
  #priority_queue for forward direction
  frontierF = [(0 + heuristic(start_node, start_node, goal_node, node_attributes), 0, start_node)]
  #priority_queue for backward direction
  frontierB = [(0 + heuristic(goal_node, start_node, goal_node, node_attributes), 0, goal_node)]
  #dictionary to store visited nodes and distance and parents in forward direction
  reachedF = {start_node : (0, None)}
  #dictionary to store visited nodes and distance and parents in backward direction
  reachedB = {goal_node : (0, None)}

  while frontierF and frontierB: #until any frontier is empty
    if frontierF[0][0] < frontierB[0][0]: #forward frontier's top's distance is less
      intersecting_node = proceed('F', adj_matrix, frontierF, reachedF, reachedB, start_node, goal_node, node_attributes)
    else: #backward frontier's top's distance is less
      intersecting_node = proceed('B', adj_matrix, frontierB, reachedB, reachedF, start_node, goal_node, node_attributes)
    if intersecting_node:
      break #node is not None, we found intersecting node

  if not intersecting_node: #goal node was never found at any depth
    return None

  #finding the path
  path = [intersecting_node]
  node = intersecting_node
  while reachedF[node][1]: #path from intersecting_node to start_node
    path.append(reachedF[node][1])
    node = reachedF[node][1]
  path.reverse() #reverse to get path from start_node to intersecting_node
  node = intersecting_node
  while reachedB[node][1]: #path from intersecting_node to goal_node
    path.append(reachedB[node][1])
    node = reachedB[node][1]
  return path


#function to find memory usage and time of execution for search algorithms
def find_resources(adj_matrix, node_attributes):
  tracemalloc.start() #start tracking memory
  start_time = time.time() #start time
  for start_node in range(len(adj_matrix)):
    for goal_node in range(start_node+1, len(adj_matrix)):
      get_ids_path(adj_matrix, start_node, goal_node) #call for all pair of nodes
  end_time = time.time() #end time
  current, peak = tracemalloc.get_traced_memory() #get memory usage
  tracemalloc.stop() #stop tracking memory
  time_elapsed = end_time - start_time
  print("\nIterative Deepening Search")
  print(f"Time Elapsed: {time_elapsed} seconds")
  print(f"Current Memory Usage: {current / 1024} KB")
  print(f"Peak Memory Usage: {peak / 1024} KB")

  tracemalloc.start() #start tracking memory
  start_time = time.time() #start time
  for start_node in range(len(adj_matrix)):
    for goal_node in range(start_node+1, len(adj_matrix)):
      get_bidirectional_search_path(adj_matrix, start_node, goal_node) #call for all pair of nodes
  end_time = time.time() #end time
  current, peak = tracemalloc.get_traced_memory() #get memory usage
  tracemalloc.stop() #stop tracking memory
  time_elapsed = end_time - start_time
  print("\nBidirectional Breadth-First Search")
  print(f"Time Elapsed: {time_elapsed} seconds")
  print(f"Current Memory Usage: {current / 1024} KB")
  print(f"Peak Memory Usage: {peak / 1024} KB")

  tracemalloc.start() #start tracking memory
  start_time = time.time() #start time
  for start_node in range(len(adj_matrix)):
    for goal_node in range(start_node+1, len(adj_matrix)):
      get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node) #call for all pair of nodes
  end_time = time.time() #end time
  current, peak = tracemalloc.get_traced_memory() #get memory usage
  tracemalloc.stop() #stop tracking memory
  time_elapsed = end_time - start_time
  print("\nA* Search")
  print(f"Time Elapsed: {time_elapsed} seconds")
  print(f"Current Memory Usage: {current / 1024} KB")
  print(f"Peak Memory Usage: {peak / 1024} KB")

  tracemalloc.start() #start tracking memory
  start_time = time.time() #start time
  for start_node in range(len(adj_matrix)):
    for goal_node in range(start_node+1, len(adj_matrix)):
      get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node) #call for all pair of nodes
  end_time = time.time() #end time
  current, peak = tracemalloc.get_traced_memory() #get memory usage
  tracemalloc.stop() #stop tracking memory
  time_elapsed = end_time - start_time
  print("\nBidirectional A* Search")
  print(f"Time Elapsed: {time_elapsed} seconds")
  print(f"Current Memory Usage: {current / 1024} KB")
  print(f"Peak Memory Usage: {peak / 1024} KB")


#function to make scatter plots
def compare_algorithms():
  #empirical results from find_resources
  algorithms = ['IDS', 'Bi-BFS', 'A*', 'Bi-A*']
  colors = ['purple', 'green', 'red', 'blue']
  times = [2327.27, 43.55, 95.19, 65.55]
  memories = [102.51, 5.86, 9.29, 8.74]
  plt.figure(figsize=(10, 6))
  for i in range(len(algorithms)):
    plt.scatter(times[i], memories[i], color=colors[i], s=20, label=algorithms[i])
  plt.title("Time vs Memory Usage")
  plt.xlabel("Time (seconds)")
  plt.ylabel("Memory (KB)")
  plt.legend(title="Algorithms")
  plt.grid(True)
  plt.show()


# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

def bonus_problem(adj_matrix):

  return []


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')

  ##the below function is used to find resource usage of all search algorithms for all pair of nodes
  # find_resources(adj_matrix, node_attributes)

  ##the below function is used to plot scatter plots to compare algorithms
  # compare_algorithms()