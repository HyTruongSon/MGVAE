import torch
import numpy as np
from sklearn.cluster import KMeans

# Discrete KL divergence
def kl_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

# +--------------------------+
# | Spectral Partitions Tree |
# +--------------------------+

class Spectral_Partitions_Tree:
    def __init__(self, adj, n_clusters, n_levels):
        self.adj = adj
        self.n_vertices = adj.shape[0]
        self.n_clusters = n_clusters
        self.n_levels = n_levels

        # Graph Laplacian
        D = np.sum(self.adj, axis = 1)
        D = np.squeeze(np.asarray(D))
        S = 1.0 / D
        self.laplacian = np.matmul(np.diag(S), np.diag(D) - self.adj) + 1e-4 * np.eye(self.n_vertices)

        # Eigen-decomposition
        w, v = np.linalg.eig(self.laplacian)
        w = np.real(w)
        v = np.real(v)
        
        # Spectral clustering
        features = np.zeros((self.n_vertices, self.n_vertices))
        features[:, :] = v[:, :]
        for i in range(self.n_vertices):
            features[:, i] /= w[i]
        if self.n_vertices >= 10:
            features = features[:, self.n_vertices-10:]

        idx = [i for i in range(self.n_vertices)]
        self.vertices = []
        self.vertices.append(idx)
        self.parent_node = []
        self.parent_node.append(-1)
        self.children_nodes = []
        self.levels = []
        N = 1
        for l in range(self.n_levels):
            start = len(self.vertices) - N
            finish = len(self.vertices)
            level = []
            for i in range(start, finish):
                level.append(i)
                idx = self.vertices[i]
                # Special case when we don't have enough vertices in this partition to cluster
                if len(idx) <= self.n_clusters:
                    children = []
                    for c in range(self.n_clusters):
                        sub_idx = idx
                        self.vertices.append(sub_idx)
                        self.parent_node.append(i)
                        children.append(len(self.vertices) - 1)
                    self.children_nodes.append(children)
                else:
                    # Apply K-Means algorithm for clustering based on vertex features
                    A = np.zeros((len(idx), features.shape[1]))
                    for j in range(len(idx)):
                        A[j, :] = features[idx[j], :]
                    kmeans = KMeans(n_clusters = self.n_clusters, random_state = 0).fit(A)
                    labels = kmeans.labels_

                    children = []
                    for c in range(self.n_clusters):
                        sub_idx = []
                        for j in range(len(labels)):
                            if labels[j] == c:
                                sub_idx.append(idx[j])
                        self.vertices.append(sub_idx)
                        self.parent_node.append(i)
                        children.append(len(self.vertices) - 1)
                    self.children_nodes.append(children)

            self.levels.append(level)
            N *= self.n_clusters

        # Bottom level
        start = len(self.vertices) - N
        finish = len(self.vertices)
        level = []
        for i in range(start, finish):
            level.append(i)
            self.children_nodes.append([])
        self.levels.append(level)

        self.n_nodes = len(self.vertices)
        print('---------------------------------')
        print('Number of nodes in the tree:', self.n_nodes)
        for i in range(self.n_nodes):
            print('Parent of node', i, ':', self.parent_node[i])
            print('Children of node', i, ':', self.children_nodes[i])
        print('---------------------------------')
        print('Number of levels:', self.n_levels)
        for l in range(self.n_levels + 1):
            print('Nodes in level', l, ':', self.levels[l])
            count = 0
            MIN = 1e9
            MAX = 0
            arr = []
            for i in self.levels[l]:
                size = len(self.vertices[i])
                count += size
                MIN = min(MIN, size)
                MAX = max(MAX, size)
                arr.append(size)
            arr = np.array(arr)
            mean = np.mean(arr)
            std = np.std(arr)
            uniform = np.array([count/len(self.levels[l]) for i in range(len(self.levels[l]))])
            print('Total number of vertices in level', l, ':', count)
            print('Minimum number of vertices in level', l, ':', MIN)
            print('Maximum number of vertices in level', l, ':', MAX)
            print('Average number of vertices in level', l, ':', mean)
            print('STD number of vertices in level', l, ':', std)
            print('KL divergence to uniform distribution in level', l, ':', kl_divergence(arr, uniform))

# +------------------------+
# | KMeans Partitions Tree |
# +------------------------+

class KMeans_Partitions_Tree:
    def __init__(self, features, n_clusters, n_levels):
        self.features = features
        self.n_vertices = features.shape[0]
        self.n_clusters = n_clusters
        self.n_levels = n_levels

        idx = [i for i in range(self.n_vertices)]
        self.vertices = []
        self.vertices.append(idx)
        self.parent_node = []
        self.parent_node.append(-1)
        self.children_nodes = []
        self.levels = []
        N = 1
        for l in range(self.n_levels):
            start = len(self.vertices) - N
            finish = len(self.vertices)
            level = []
            for i in range(start, finish):
                level.append(i)
                idx = self.vertices[i]
                # Special case when we don't have enough vertices in this partition to cluster
                if len(idx) <= self.n_clusters:
                    children = []
                    for c in range(self.n_clusters):
                        sub_idx = idx
                        self.vertices.append(sub_idx)
                        self.parent_node.append(i)
                        children.append(len(self.vertices) - 1)
                    self.children_nodes.append(children)
                else:
                    # Apply K-Means algorithm for clustering based on vertex features
                    A = np.zeros((len(idx), features.shape[1]))
                    for j in range(len(idx)):
                        A[j, :] = features[idx[j], :]
                    kmeans = KMeans(n_clusters = self.n_clusters, random_state = 0).fit(A)
                    labels = kmeans.labels_

                    children = []
                    for c in range(self.n_clusters):
                        sub_idx = []
                        for j in range(len(labels)):
                            if labels[j] == c:
                                sub_idx.append(idx[j])
                        self.vertices.append(sub_idx)
                        self.parent_node.append(i)
                        children.append(len(self.vertices) - 1)
                    self.children_nodes.append(children)

            self.levels.append(level)
            N *= self.n_clusters

        # Bottom level
        start = len(self.vertices) - N
        finish = len(self.vertices)
        level = []
        for i in range(start, finish):
            level.append(i)
            self.children_nodes.append([])
        self.levels.append(level)

        self.n_nodes = len(self.vertices)
        print('---------------------------------')
        print('Number of nodes in the tree:', self.n_nodes)
        for i in range(self.n_nodes):
            print('Parent of node', i, ':', self.parent_node[i])
            print('Children of node', i, ':', self.children_nodes[i])
        print('---------------------------------')
        print('Number of levels:', self.n_levels)
        for l in range(self.n_levels + 1):
            print('Nodes in level', l, ':', self.levels[l])
            count = 0
            MIN = 1e9
            MAX = 0
            arr = []
            for i in self.levels[l]:
                size = len(self.vertices[i])
                count += size
                MIN = min(MIN, size)
                MAX = max(MAX, size)
                arr.append(size)
            arr = np.array(arr)
            mean = np.mean(arr)
            std = np.std(arr)
            uniform = np.array([count/len(self.levels[l]) for i in range(len(self.levels[l]))])
            print('Total number of vertices in level', l, ':', count)
            print('Minimum number of vertices in level', l, ':', MIN)
            print('Maximum number of vertices in level', l, ':', MAX)
            print('Average number of vertices in level', l, ':', mean)
            print('STD number of vertices in level', l, ':', std)
            print('KL divergence to uniform distribution in level', l, ':', kl_divergence(arr, uniform))

# +-------------------------------+
# | Multiresolution Graph Targets |
# +-------------------------------+

class Multiresolution_Graph_Targets:
    def __init__(self, adj, tree, device = 'cpu'):
        self.adj = adj
        self.tree = tree
        self.n_levels = tree.n_levels
        self.device = device

        self.local_targets = []
        self.global_target = []
        for l in range(self.n_levels + 1):
            self.local_targets.append([])
            self.global_target.append([])

        l = self.n_levels
        while l >= 0:
            if l == self.n_levels:
                # Local targets
                for i in tree.levels[l]:
                    A = self.adj[tree.vertices[i], :]
                    A = A[:, tree.vertices[i]]
                    N = len(tree.vertices[i])
                    assert A.shape[0] == N
                    assert A.shape[1] == N
                    target = np.zeros((N, N))
                    target[:, :] = A[:, :]
                    self.local_targets[l].append(torch.from_numpy(target).type(torch.FloatTensor).to(device = device))
            else:
                # Local targets
                for i in tree.levels[l]:
                    K = len(tree.children_nodes[i])
                    target = np.zeros((K, K))
                    for uu in range(K):
                        u = tree.children_nodes[i][uu]
                        for vv in range(K):
                            v = tree.children_nodes[i][vv]
                            A = self.adj[tree.vertices[u], :]
                            A = A[:, tree.vertices[v]]
                            target[uu, vv] = np.sum(A)
                    target /= np.max(target)
                    self.local_targets[l].append(torch.from_numpy(target).type(torch.FloatTensor).to(device = device))

            # Global target
            M = len(tree.levels[l])
            target = np.zeros((M, M))
            for ii in range(M):
                i = tree.levels[l][ii]
                for jj in range(M):
                    j = tree.levels[l][jj]
                    A = self.adj[tree.vertices[i], :]
                    A = A[:, tree.vertices[j]]
                    target[ii, jj] = np.sum(A)
            target /= np.max(target)
            self.global_target[l] = torch.from_numpy(target).type(torch.FloatTensor).to(device = device)
            
            l -= 1

        for l in range(self.n_levels + 1):
            print('---------------------------------')
            print('Level', l)
            if len(self.local_targets[l]) > 0:
                print('Local targets =')
                for element in self.local_targets[l]:
                    print(element.size())
            print('Global target =', self.global_target[l].size())
            print(self.global_target[l])
