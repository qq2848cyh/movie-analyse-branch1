import networkx as nx
import community as community_louvain
import numpy as np
from itertools import combinations
from typing import List, Dict, Optional
from collections import Counter, defaultdict


class NetworkAnalyzer:
    """演员-导演协作网络分析器"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.graph: Optional[nx.Graph] = None
        self.actor_graph: Optional[nx.Graph] = None
        self.director_graph: Optional[nx.Graph] = None

    def build(self):
        c = self.db._get_conn()

        c.execute("DROP TABLE IF EXISTS net_actor_stats")
        c.execute("""
            CREATE TEMP TABLE net_actor_stats AS
            SELECT ma.person_id, COUNT(DISTINCT ma.movie_id) as movie_count
            FROM movie_actors ma
            INNER JOIN movies_valid mv ON ma.movie_id = mv.movie_id
            GROUP BY ma.person_id
        """)
        min_count = 3

        rows = c.execute("""
            SELECT ma.movie_id, ma.person_id, ma.actor_name
            FROM movie_actors ma
            INNER JOIN net_actor_stats ns ON ma.person_id = ns.person_id
            WHERE ns.movie_count >= ?
            ORDER BY ma.movie_id
        """, (min_count,)).fetchall()

        movie_actors = defaultdict(list)
        for r in rows:
            movie_actors[r["movie_id"]].append((r["person_id"], r["actor_name"]))

        self.actor_graph = nx.Graph()
        edge_weights = defaultdict(int)
        node_names = {}

        for movie_id, actors in movie_actors.items():
            if len(actors) < 2:
                continue
            if len(actors) > 10:
                 actors = actors[:10]
            for pid, name in actors:
                node_names[pid] = name
            for (u, uname), (v, vname) in combinations(actors, 2):
                key = (u, v) if u < v else (v, u)
                edge_weights[key] += 1

        self.actor_graph.add_nodes_from((pid, {"name": name}) for pid, name in node_names.items())
        self.actor_graph.add_weighted_edges_from(
            (u, v, w) for (u, v), w in edge_weights.items()
        )

        self.director_graph = nx.Graph()
        d_rows = c.execute("""
            SELECT md.movie_id, md.person_id, md.director_name
            FROM movie_directors md
            INNER JOIN movies_valid mv ON md.movie_id = mv.movie_id
            ORDER BY md.movie_id
        """).fetchall()

        d_edge_weights = defaultdict(int)
        d_node_types = {}

        for r in d_rows:
            movie_id = r["movie_id"]
            actors = movie_actors.get(movie_id, [])
            if len(actors) > 10:
                actors = actors[:10]
            d_node_types[r["person_id"]] = (r["director_name"], "director")
            for a_pid, a_name in actors:
                d_node_types[a_pid] = (a_name, "actor")
                key = (r["person_id"], a_pid) if r["person_id"] < a_pid else (a_pid, r["person_id"])
                d_edge_weights[key] += 1

        self.director_graph.add_nodes_from(
            (pid, {"name": name, "type": t}) for pid, (name, t) in d_node_types.items()
        )
        self.director_graph.add_weighted_edges_from(
            (u, v, w) for (u, v), w in d_edge_weights.items()
        )

        self.graph = self.actor_graph

        return {
            "actor_nodes": self.actor_graph.number_of_nodes(),
            "actor_edges": self.actor_graph.number_of_edges(),
        }

    def get_global_stats(self) -> Dict:
        G = self.graph
        n = G.number_of_nodes()
        e = G.number_of_edges()
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        avg_degree = 2 * e / n if n > 0 else 0
        import random
        avg_clustering = nx.average_clustering(G, nodes=random.sample(
            list(G.nodes()), min(800, n)
        ))

        components = list(nx.connected_components(G))
        num_components = len(components)
        largest = G.subgraph(max(components, key=len))
        l_n = largest.number_of_nodes()

        if l_n > 1:
            sample_nodes = random.sample(list(largest.nodes()), min(200, l_n))
            path_lengths = []
            for s in sample_nodes:
                lengths = nx.single_source_shortest_path_length(largest, s, cutoff=15)
                path_lengths.extend(lengths.values())
            avg_path = np.mean(path_lengths) if path_lengths else 0

            ecc = {}
            for s in random.sample(list(largest.nodes()), min(100, l_n)):
                dists = nx.single_source_shortest_path_length(largest, s, cutoff=30)
                if dists:
                    ecc[s] = max(dists.values())
            diameter = max(ecc.values()) if ecc else 0
        else:
            avg_path = 0
            diameter = 0

        r_clustering = density
        r_path = np.log(n) / np.log(avg_degree) if avg_degree > 1 else n

        return {
            "nodes": n,
            "edges": e,
            "density": round(density, 6),
            "avg_degree": round(avg_degree, 2),
            "avg_path": round(avg_path, 4),
            "diameter": diameter,
            "clustering": round(avg_clustering, 4),
            "random_path": round(r_path, 4),
            "random_clustering": round(r_clustering, 6),
            "is_small_world": bool(avg_clustering > r_clustering * 5 and avg_path < r_path * 2
                              if avg_path > 0 else False),
            "components": num_components,
            "largest_component": l_n,
        }

    def get_degree_distribution(self) -> List[Dict]:
        """度分布"""
        degrees = [d for _, d in self.graph.degree()]
        counter = Counter(degrees)
        dist = sorted(counter.items())

        log_x = np.log([d[0] for d in dist if d[0] > 0])
        log_y = np.log([d[1] for d in dist if d[0] > 0])
        slope, intercept = np.polyfit(log_x, log_y, 1)

        return {
            "distribution": [
                {"degree": k, "count": v} for k, v in dist if k > 0
            ],
            "powerlaw_exponent": round(-slope, 4),
            "powerlaw_fit": f"P(k) ∝ k^(-{-slope:.2f})",
        }

    def get_centrality(self, top_n: int = 20) -> Dict:
        """中心性排名"""
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(largest_cc, k=min(200, len(largest_cc)))

        deg_rank = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        bet_rank = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return {
            "degree": [
                {
                    "id": pid,
                    "name": G.nodes[pid].get("name", str(pid)),
                    "value": round(v, 6),
                }
                for pid, v in deg_rank
            ],
            "betweenness": [
                {
                    "id": pid,
                    "name": G.nodes[pid].get("name", str(pid)),
                    "value": round(v, 6),
                }
                for pid, v in bet_rank
            ],
        }

    def get_communities(self) -> Dict:
        """社区检测"""
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        partition = community_louvain.best_partition(largest_cc)
        community_sizes = Counter(partition.values())

        node_community = {}
        for node, cid in partition.items():
            node_community[int(node)] = int(cid)

        return {
            "num_communities": len(community_sizes),
            "community_sizes": sorted(
                [{"id": k, "size": v} for k, v in community_sizes.items()],
                key=lambda x: x["size"],
                reverse=True,
            )[:10],
            "modularity": round(
                community_louvain.modularity(partition, largest_cc), 4
            ),
            "node_community": node_community,
        }

    def get_force_graph_data(self, max_nodes: int = 200) -> Dict:
        """力导向图数据"""
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        deg_cent = nx.degree_centrality(largest_cc)
        top_nodes = sorted(deg_cent, key=deg_cent.get, reverse=True)[:max_nodes]
        sub = largest_cc.subgraph(top_nodes)

        partition = community_louvain.best_partition(sub)
        communities = {}
        for node, cid in partition.items():
            communities.setdefault(cid, []).append(node)

        nodes = []
        for node in sub.nodes():
            deg = sub.degree(node)
            size = max(6, min(30, 4 + np.log(deg + 1) * 6))
            nodes.append({
                "id": int(node),
                "name": sub.nodes[node].get("name", str(node)),
                "symbolSize": round(size, 1),
                "category": int(partition[node]),
            })

        links = []
        for u, v in sub.edges():
            links.append({
                "source": int(u),
                "target": int(v),
            })

        return {
            "nodes": nodes,
            "links": links,
            "categories": [
                {"name": f"社区 {i}"} for i in range(len(communities))
            ],
        }

    def get_all_stats(self) -> Dict:
        """汇总所有分析结果"""
        if self.graph is None:
            self.build()

        return {
            "build_info": {
                "actor_nodes": self.actor_graph.number_of_nodes(),
                "actor_edges": self.actor_graph.number_of_edges(),
                "director_nodes": self.director_graph.number_of_nodes(),
                "director_edges": self.director_graph.number_of_edges(),
            },
            "global_stats": self.get_global_stats(),
            "degree_distribution": self.get_degree_distribution(),
            "centrality": self.get_centrality(),
            "communities": self.get_communities(),
            "force_graph": self.get_force_graph_data(),
        }
