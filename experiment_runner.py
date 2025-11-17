import os
import time
import tracemalloc
import heapq
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USE_OSM = True
try:
    import osmnx as ox
    import networkx as nx
    ox.settings.use_cache = True
    ox.settings.log_console = False
except Exception as e:
    print("Erro ao importar OSMnx ou NetworkX:", e)
    print("Instale 'osmnx==2.0.6' e 'networkx' antes de rodar.")
    USE_OSM = False
    raise

sns.set(style="whitegrid")
random.seed(42)

def load_graph_ipsep(place_name="Ipsep, Recife, Pernambuco, Brazil", network_type="drive"):

    print("Baixando grafo OSM para:", place_name)

    G = ox.graph.graph_from_place(place_name, network_type=network_type)

    print("Projetando grafo...")
    G = ox.project_graph(G) 

    for u, v, k, data in G.edges(keys=True, data=True):
        data["weight"] = data.get("length", data.get("weight", 1.0))

    print(f"Grafo pronto: {len(G.nodes())} nós, {len(G.edges())} arestas.")
    return G

def edge_weight(u, v, G):
    data = G.get_edge_data(u, v)
    if isinstance(data, dict):
        data = next(iter(data.values()))
    return data.get("weight", 1.0)

def path_length(G, path):
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += edge_weight(a, b, G)
    return total

from collections import deque

def bfs_instrumented(G, source, target):
    visited = {source}
    prev = {source: None}
    q = deque([source])
    nodes_expanded = 0
    while q:
        u = q.popleft()
        nodes_expanded += 1
        if u == target:
            break
        for v in G.neighbors(u):
            if v not in visited:
                visited.add(v)
                prev[v] = u
                q.append(v)
    if target not in prev:
        return None, math.inf, nodes_expanded
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    dist = path_length(G, path)
    return path, dist, nodes_expanded

def dijkstra_instrumented(G, source, target):
    dist = {source: 0.0}
    prev = {}
    pq = [(0.0, source)]
    visited = set()
    nodes_expanded = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        nodes_expanded += 1
        if u == target:
            break
        for v in G.neighbors(u):
            w = edge_weight(u, v, G)
            nd = d + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return None, math.inf, nodes_expanded
    path = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path, dist[target], nodes_expanded

def euclidean_heuristic(u, v, G):
    ux, uy = G.nodes[u]['x'], G.nodes[u]['y']
    vx, vy = G.nodes[v]['x'], G.nodes[v]['y']
    return math.hypot(ux - vx, uy - vy)

def astar_instrumented(G, source, target):
    gscore = {source: 0.0}
    fscore = {source: euclidean_heuristic(source, target, G)}
    pq = [(fscore[source], source)]
    prev = {}
    visited = set()
    nodes_expanded = 0
    while pq:
        f, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        nodes_expanded += 1
        if u == target:
            break
        for v in G.neighbors(u):
            tentative_g = gscore[u] + edge_weight(u, v, G)
            if v not in gscore or tentative_g < gscore[v]:
                gscore[v] = tentative_g
                prev[v] = u
                fscore_v = tentative_g + euclidean_heuristic(v, target, G)
                heapq.heappush(pq, (fscore_v, v))
    if target not in gscore:
        return None, math.inf, nodes_expanded
    path = []
    cur = target
    while cur != source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path, gscore[target], nodes_expanded


def measure_run(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = func(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = t1 - t0
    return res, elapsed, peak


def get_pois_nodes(G, place_name="Ipsep, Recife, Pernambuco, Brazil"):

    print("Buscando POIs (hospital/clinic/school) para:", place_name)

    tags = {"amenity": ["hospital", "clinic", "school"]}


    pois = ox.features.features_from_place(place_name, tags=tags)

    if pois is None or pois.empty:
        print("Nenhum POI encontrado. Retornando listas vazias.")
        return [], []

    pois = pois[pois.geometry.type == "Point"]

    hospitals = pois[pois["amenity"] == "hospital"]
    clinics = pois[pois["amenity"] == "clinic"]
    schools = pois[pois["amenity"] == "school"]

    hospital_nodes = []
    school_nodes = []

    for geom in pd.concat([hospitals.geometry, clinics.geometry]):
        try:
            node = ox.distance.nearest_nodes(G, X=geom.x, Y=geom.y)
            hospital_nodes.append(node)
        except:
            pass

    for geom in schools.geometry:
        try:
            node = ox.distance.nearest_nodes(G, X=geom.x, Y=geom.y)
            school_nodes.append(node)
        except:
            pass

    return hospital_nodes, school_nodes


def run_experiments(G, od_pairs, out_csv="results_ipsep.csv"):
    rows = []
    for idx, (o, d) in enumerate(od_pairs):
        
        (path_b, dist_b, nodes_b), tb, mb = measure_run(bfs_instrumented, G, o, d)
        
        (path_d, dist_d, nodes_d), td, md = measure_run(dijkstra_instrumented, G, o, d)
        
        (path_a, dist_a, nodes_a), ta, ma = measure_run(astar_instrumented, G, o, d)

        def safe(x): return float("nan") if x is None else x

        row = {
            "pair_idx": idx,
            "origin": o, "dest": d,
            "bfs_time_s": tb, "bfs_mem_peak_B": mb, "bfs_nodes_expanded": nodes_b, "bfs_dist_m": safe(dist_b),
            "dij_time_s": td, "dij_mem_peak_B": md, "dij_nodes_expanded": nodes_d, "dij_dist_m": safe(dist_d),
            "astar_time_s": ta, "astar_mem_peak_B": ma, "astar_nodes_expanded": nodes_a, "astar_dist_m": safe(dist_a),
        }

        row["bfs_subopt"] = row["bfs_dist_m"] / row["dij_dist_m"] if not math.isinf(row["dij_dist_m"]) else None
        row["astar_subopt"] = row["astar_dist_m"] / row["dij_dist_m"] if not math.isinf(row["dij_dist_m"]) else None

        rows.append(row)
        print(f"Par {idx+1}/{len(od_pairs)} concluído.")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def plot_summary(df, out_prefix="figures/ipsep"):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    plt.figure(figsize=(8,4))
    df_time = df[["bfs_time_s","dij_time_s","astar_time_s"]].melt(var_name="alg", value_name="time_s")
    sns.boxplot(x="alg", y="time_s", data=df_time)
    plt.title("Tempo por algoritmo (s)")
    plt.savefig(out_prefix + "_time_box.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    df_nodes = df[["bfs_nodes_expanded","dij_nodes_expanded","astar_nodes_expanded"]].melt(var_name="alg", value_name="nodes")
    sns.boxplot(x="alg", y="nodes", data=df_nodes)
    plt.title("Nós expandidos por algoritmo")
    plt.savefig(out_prefix + "_nodes_box.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8,4))
    df_sub = df[["bfs_subopt","astar_subopt"]].melt(var_name="alg", value_name="subopt")
    sns.boxplot(x="alg", y="subopt", data=df_sub)
    plt.title("Subótimo relativo (1.0 = ótimo)")
    plt.ylim(0.9, 2.0)
    plt.savefig(out_prefix + "_subopt_box.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    place = "Ipsep, Recife, Pernambuco, Brazil"
    G = load_graph_ipsep(place)

    hospital_nodes, school_nodes = get_pois_nodes(G, place)

    if len(hospital_nodes) < 3 or len(school_nodes) < 3:
        print("POIs insuficientes na área IPSEP; usando nós aleatórios.")
        nodes = list(G.nodes())
        random.shuffle(nodes)
        hospital_nodes = nodes[:5]
        school_nodes = nodes[5:10]

    od_pairs = []
    n_pairs = min(10, len(hospital_nodes)*len(school_nodes))

    for i in range(n_pairs):
        o = hospital_nodes[i % len(hospital_nodes)]
        d = school_nodes[i % len(school_nodes)]
        if o != d:
            od_pairs.append((o, d))

    print(f"Usando {len(od_pairs)} pares O-D.")

    df = run_experiments(G, od_pairs, out_csv="results_ipsep.csv")
    print("Resultados salvos em results_ipsep.csv")

    plot_summary(df, out_prefix="figures/ipsep")
    print("Figuras salvas em ./figures/")
