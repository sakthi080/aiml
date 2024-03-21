import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
cities = {'A': 'SEACET', 'B': 'K.R.Puram', 'C': 'ITPL', 'D': 'RMnagar', 'E': 'TinFactory'}
roads = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
G.add_nodes_from(cities.keys())
G.add_edges_from(roads)
pos = nx.spring_layout(G)
nx.draw(G, pos, labels=cities, with_labels=True, font_weight='bold',
        node_size=780, node_color='skyblue', font_size=10, edge_color='black')
plt.title('City Map')
plt.show()
def bfs(graph, start):
    visited = set()
    queue = [start]
    bfs_result = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            bfs_result.append(node)
            visited.add(node)
            queue.extend(neighbor for neighbor in graph.neighbors(node) if neighbor not in visited)
    return bfs_result
def dfs(graph, start):
    visited = set()
    dfs_result = []

    def dfs_recursive(node):
        visited.add(node)
        dfs_result.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs_recursive(neighbor)

    dfs_recursive(start)
    return dfs_result
start_city = 'A'
bfs_result = bfs(G, start_city)
dfs_result = dfs(G, start_city)
print('BFS Result:', [cities[node] for node in bfs_result])
print('DFS Result:', [cities[node] for node in dfs_result])
