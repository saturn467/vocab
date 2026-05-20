class ClusterBalancer:

    def __init__(self, target_size=15, min_size=10, max_size=18):

        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size


    def balance(self, clusters, terms):

        balanced_clusters = []

        for cluster_indices in clusters.values():

            cluster_terms = [terms[i] for i in cluster_indices]

            size = len(cluster_terms)


            # if cluster is too large → split it
            if size > self.max_size:

                for i in range(0, size, self.target_size):

                    balanced_clusters.append(
                        cluster_terms[i:i+self.target_size]
                    )


            # if cluster size is acceptable
            elif size >= self.min_size:

                balanced_clusters.append(cluster_terms)


            # if cluster too small → keep temporarily
            else:

                balanced_clusters.append(cluster_terms)


        return balanced_clusters