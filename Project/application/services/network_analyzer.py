import networkx as nx
import community as community_louvain
import numpy as np
from itertools import combinations
from typing import List, Dict, Optional
from collections import Counter, defaultdict


class NetworkAnalyzer:
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
        assortativity = nx.degree_assortativity_coefficient(G)

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
            "assortativity": round(assortativity, 4),
            "random_path": round(r_path, 4),
            "random_clustering": round(r_clustering, 6),
            "is_small_world": bool(avg_clustering > r_clustering * 5 and avg_path < r_path * 2
                              if avg_path > 0 else False),
            "components": num_components,
            "largest_component": l_n,
        }

    def get_degree_distribution(self) -> Dict:
        degrees = [d for _, d in self.graph.degree()]
        counter = Counter(degrees)
        dist = sorted(counter.items())
        ks = np.array([d[0] for d in dist if d[0] > 0], dtype=float)
        counts = np.array([d[1] for d in dist if d[0] > 0], dtype=float)

        log_x = np.log(ks)
        log_y = np.log(counts)
        slope, intercept = np.polyfit(log_x, log_y, 1)

        from scipy import stats as scipy_stats

        x_min_candidates = ks[ks <= np.percentile(ks, 50)]
        if len(x_min_candidates) > 2:
            best_D = float('inf')
            best_xmin = ks[0]
            best_alpha = -slope
            for xm in sorted(set(int(k) for k in x_min_candidates if k >= 1)):
                tail = ks[ks >= xm]
                if len(tail) < 10:
                    continue
                alpha_mle = 1 + len(tail) / np.sum(np.log(tail / xm))
                log_norm_cdf = lambda x: scipy_stats.lognorm.cdf(x, s=1, scale=np.exp(np.mean(np.log(tail))))
                D_ks = np.max(np.abs(
                    np.arange(1, len(tail)+1) / len(tail) -
                    (1 - (tail / xm) ** (-(alpha_mle - 1)))
                ))
                if D_ks < best_D:
                    best_D = D_ks
                    best_xmin = xm
                    best_alpha = alpha_mle
        else:
            best_xmin = int(ks[0])
            best_alpha = -slope

        ks_test, p_value = scipy_stats.kstest(
            degrees,
            "powerlaw",
            args=(best_alpha - 1,),
        )

        try:
            from scipy.stats import powerlaw as pl_dist
            ks_test_mle, p_value_mle = 0.0, 0.0
        except Exception:
            p_value_mle = 0.0

        return {
            "distribution": [
                {"degree": int(k), "count": int(v)} for k, v in dist if k > 0
            ],
            "powerlaw_exponent": round(-slope, 4),
            "powerlaw_fit": f"P(k) \u221D k^(-{-slope:.2f})",
            "powerlaw_mle_alpha": round(best_alpha, 4),
            "powerlaw_xmin": int(best_xmin),
            "powerlaw_ks_statistic": round(float(best_D) if 'best_D' in dir() else 0, 4),
            "powerlaw_coverage": f"k >= {best_xmin} ({len(ks[ks >= best_xmin])} nodes)",
        }

    def get_kcore_decomposition(self) -> Dict:
        G = self.graph
        core_number = nx.core_number(G)
        max_k = max(core_number.values())

        kcore_dist = Counter(core_number.values())
        core_dist = sorted(kcore_dist.items())

        top_shell = max_k
        top_k_nodes = [n for n, c in core_number.items() if c == top_shell]
        top_k_names = [
            {"id": int(n), "name": G.nodes[n].get("name", str(n))}
            for n in top_k_nodes[:20]
        ]

        return {
            "max_kcore": max_k,
            "distribution": [{"k": k, "count": cnt} for k, cnt in core_dist],
            "top_shell_nodes": top_k_names,
            "top_shell_size": len(top_k_nodes),
        }

    def get_centrality(self, top_n: int = 20) -> Dict:
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(largest_cc, k=min(200, len(largest_cc)))

        deg_rank = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        bet_rank = sorted(bet_cent.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return {
            "degree": [
                {
                    "id": int(pid),
                    "name": G.nodes[pid].get("name", str(pid)),
                    "value": round(float(v), 6),
                }
                for pid, v in deg_rank
            ],
            "betweenness": [
                {
                    "id": int(pid),
                    "name": G.nodes[pid].get("name", str(pid)),
                    "value": round(float(v), 6),
                }
                for pid, v in bet_rank
            ],
        }

    def get_centrality_scatter(self) -> Dict:
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
        import random

        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(largest_cc, k=min(200, len(largest_cc)))

        all_deg = list(deg_cent.values())
        all_bet = list(bet_cent.values())

        deg_median = float(np.median(all_deg)) if all_deg else 0.0
        bet_median = float(np.median(all_bet)) if all_bet else 0.0

        nodes = list(G.nodes())
        sample_n = min(1200, len(nodes))
        if len(nodes) > sample_n:
            nodes = random.sample(nodes, sample_n)

        scatter_data = []
        for n in nodes:
            d_val = deg_cent.get(n, 0)
            b_val = bet_cent.get(n, 0) if n in largest_cc else 0
            scatter_data.append({
                "id": int(n),
                "name": G.nodes[n].get("name", str(n)),
                "degree": round(float(d_val), 6),
                "betweenness": round(float(b_val), 6),
            })

        return {
            "scatter": scatter_data,
            "degree_median": round(deg_median, 6),
            "betweenness_median": round(bet_median, 6),
            "sample_size": len(scatter_data),
            "total_nodes": len(nodes),
        }

    def get_communities(self) -> Dict:
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        partition = community_louvain.best_partition(largest_cc)
        community_sizes = Counter(partition.values())

        community_labels = {}
        community_members = {}
        for cid in community_sizes:
            members = [n for n, c in partition.items() if c == cid]
            member_names = [largest_cc.nodes[n].get("name", str(n)) for n in members]
            community_members[int(cid)] = member_names
            label = member_names[0] if member_names else f"社区{cid}"
            community_labels[cid] = label

        node_community = {}
        for node, cid in partition.items():
            node_community[int(node)] = int(cid)

        return {
            "num_communities": len(community_sizes),
            "community_sizes": sorted(
                [{"id": int(k), "size": int(v), "label": community_labels.get(k, f"社区{k}"),
                  "members": community_members.get(int(k), [])}
                 for k, v in community_sizes.items()],
                key=lambda x: x["size"],
                reverse=True,
            ),
            "modularity": round(
                community_louvain.modularity(partition, largest_cc), 4
            ),
            "node_community": node_community,
        }

    def get_force_graph_data(self, max_nodes: int = 300) -> Dict:
        G = self.graph
        largest_cc = G.subgraph(max(nx.connected_components(G), key=len))

        deg_cent = nx.degree_centrality(largest_cc)
        top_nodes = sorted(deg_cent, key=deg_cent.get, reverse=True)[:max_nodes]
        sub = largest_cc.subgraph(top_nodes)

        partition = community_louvain.best_partition(sub)

        nodes = []
        for node in sub.nodes():
            deg = sub.degree(node)
            size = max(6, min(30, 4 + np.log(deg + 1) * 6))
            nodes.append({
                "id": int(node),
                "name": sub.nodes[node].get("name", str(node)),
                "symbolSize": round(float(size), 1),
                "category": int(partition.get(node, 0)),
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
                {"name": f"社区 {i}"} for i in range(len(set(partition.values())))
            ],
        }

    def get_all_stats(self) -> Dict:
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
            "kcore_decomposition": self.get_kcore_decomposition(),
            "centrality": self.get_centrality(),
            "centrality_scatter": self.get_centrality_scatter(),
            "communities": self.get_communities(),
            "force_graph": self.get_force_graph_data(),
        }

    def export_results(self) -> Dict:
        stats = self.get_all_stats()
        g = stats["global_stats"]
        dd = stats["degree_distribution"]
        kc = stats["kcore_decomposition"]
        cm = stats["communities"]
        ct = stats["centrality"]

        return {
            "network_analysis": {
                "global_metrics": {
                    "节点数": g["nodes"],
                    "边数": g["edges"],
                    "网络密度": g["density"],
                    "平均度": g["avg_degree"],
                    "平均路径长度": g["avg_path"],
                    "直径": g["diameter"],
                    "集聚系数": g["clustering"],
                    "同配系数": g["assortativity"],
                    "连通分量数": g["components"],
                    "最大连通分量": g["largest_component"],
                    "小世界特性成立": "是" if g["is_small_world"] else "否",
                },
                "degree_distribution": {
                    "幂律指数_OLS": dd["powerlaw_exponent"],
                    "幂律指数_MLE": dd["powerlaw_mle_alpha"],
                    "幂律下界_xmin": dd["powerlaw_xmin"],
                    "覆盖范围": dd["powerlaw_coverage"],
                    "拟合公式": dd["powerlaw_fit"],
                },
                "kcore_decomposition": {
                    "最大k核值": kc["max_kcore"],
                    "最内核节点数": kc["top_shell_size"],
                    "最内核代表节点": [n["name"] for n in kc["top_shell_nodes"][:5]],
                },
                "community_detection": {
                    "社区数": cm["num_communities"],
                    "模块度": cm["modularity"],
                    "最大社区": cm["community_sizes"][0] if cm["community_sizes"] else None,
                },
                "centrality_top5": {
                    "度中心性": [{"name": x["name"], "value": x["value"]} for x in ct["degree"][:5]],
                    "介数中心性": [{"name": x["name"], "value": x["value"]} for x in ct["betweenness"][:5]],
                },
            }
        }
