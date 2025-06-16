import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Generate sample data/
np.random.seed(42)
X, y_true = make_blobs(n_samples=6000, centers=6, cluster_std=1, random_state=42)


class ImprovedMolecularDPMM:
    """
    Покращена версія з зворотним проходом для оптимізації кількості кластерів
    """

    def __init__(self, alpha=1.0, max_iter=12, min_cluster_size=5,
                 initial_split_threshold=0.3, refinement_threshold=0.1,
                 max_clusters=40, use_smart_init=True,
                 backward_optimization=True, target_clusters=None):
        """
        Parameters:
        -----------
        backward_optimization : bool
            Використовувати зворотний прохід для оптимізації
        target_clusters : int or None
            Цільова кількість кластерів (якщо відома)
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.initial_split_threshold = initial_split_threshold
        self.refinement_threshold = refinement_threshold
        self.max_clusters = max_clusters
        self.use_smart_init = use_smart_init
        self.backward_optimization = backward_optimization
        self.target_clusters = target_clusters

        # Внутрішні параметри
        self.clusters = []
        self.n_clusters = 0
        self.converged = False
        self.quality_history = []
        self.backward_history = []
        self.best_configuration = None
        self.best_score = -np.inf

    def _calculate_cluster_spread(self, cluster_data):
        """Розрахунок розкиданості точок в кластері"""
        if len(cluster_data) < 4:
            return 0.0

        center = np.mean(cluster_data, axis=0)
        distances = np.linalg.norm(cluster_data - center, axis=1)
        cv = np.std(distances) / (np.mean(distances) + 1e-8)

        if cluster_data.shape[1] > 1:
            pca = PCA(n_components=2)
            pca.fit(cluster_data)
            explained_ratio = pca.explained_variance_ratio_
            elongation = explained_ratio[0] - explained_ratio[1] if len(explained_ratio) > 1 else explained_ratio[0]
        else:
            elongation = 0.5

        spread_score = cv * 0.7 + elongation * 0.3
        return spread_score

    def _smart_initialization(self, X):
        """Розумна ініціалізація з K-means"""
        print("  Using smart initialization with K-means...")

        best_k = 1
        best_score = -np.inf

        for k in range(2, min(8, len(X) // 50)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue

        if best_k > 1:
            print(f"    → Initializing with {best_k} clusters (silhouette: {best_score:.3f})")
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            self.clusters = []
            for k in range(best_k):
                cluster_data = X[labels == k]
                if len(cluster_data) > 0:
                    cluster = {
                        'data': cluster_data,
                        'mean': np.mean(cluster_data, axis=0),
                        'cov': np.cov(cluster_data.T) + np.eye(cluster_data.shape[1]) * 1e-6,
                        'weight': len(cluster_data) / len(X),
                        'size': len(cluster_data),
                        'id': len(self.clusters)
                    }
                    self.clusters.append(cluster)

            self.n_clusters = len(self.clusters)
            return True

        return False

    def _initialize_clusters(self, X):
        """Ініціалізація кластерів"""
        if self.use_smart_init:
            if self._smart_initialization(X):
                return

        print("  Using single cluster initialization...")
        cluster = {
            'data': X.copy(),
            'mean': np.mean(X, axis=0),
            'cov': np.cov(X.T) + np.eye(X.shape[1]) * 1e-6,
            'weight': 1.0,
            'size': len(X),
            'id': 0
        }
        self.clusters = [cluster]
        self.n_clusters = 1

    def _enhanced_split_analysis(self, cluster_data):
        """Покращений аналіз можливості поділу"""
        if len(cluster_data) < 6:
            return None, 0.0

        best_split = None
        best_score = 0.0

        # PCA-based поділ
        pca = PCA(n_components=min(cluster_data.shape[1], 2))
        pca.fit(cluster_data)

        for comp_idx in range(pca.n_components_):
            projections = pca.transform(cluster_data)[:, comp_idx]

            for percentile in [30, 40, 50, 60, 70]:
                threshold = np.percentile(projections, percentile)
                mask1 = projections <= threshold
                mask2 = projections > threshold

                if np.sum(mask1) >= 3 and np.sum(mask2) >= 3:
                    part1, part2 = cluster_data[mask1], cluster_data[mask2]
                    score = self._evaluate_split_quality(cluster_data, part1, part2)

                    if score > best_score:
                        best_score = score
                        best_split = (part1, part2)

        # K-means поділ
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            split_labels = kmeans.fit_predict(cluster_data)

            part1 = cluster_data[split_labels == 0]
            part2 = cluster_data[split_labels == 1]

            if len(part1) >= 3 and len(part2) >= 3:
                score = self._evaluate_split_quality(cluster_data, part1, part2)
                if score > best_score:
                    best_score = score
                    best_split = (part1, part2)
        except:
            pass

        return best_split, best_score

    def _evaluate_split_quality(self, original, part1, part2):
        """Комплексна оцінка якості поділу"""
        if len(part1) < 3 or len(part2) < 3:
            return 0.0

        def compactness(data):
            if len(data) < 2:
                return 0
            center = np.mean(data, axis=0)
            return np.mean(np.linalg.norm(data - center, axis=1))

        original_compactness = compactness(original)
        new_compactness = (len(part1) * compactness(part1) + len(part2) * compactness(part2)) / len(original)
        compactness_improvement = (original_compactness - new_compactness) / (original_compactness + 1e-8)

        center1 = np.mean(part1, axis=0)
        center2 = np.mean(part2, axis=0)
        separation = np.linalg.norm(center1 - center2)
        avg_size = (compactness(part1) + compactness(part2)) / 2
        separation_score = separation / (avg_size + 1e-8)

        size_ratio = min(len(part1), len(part2)) / max(len(part1), len(part2))
        balance_score = size_ratio

        try:
            from scipy.stats import ttest_ind
            if original.shape[1] == 1:
                _, p_value = ttest_ind(part1.ravel(), part2.ravel())
            else:
                pca = PCA(n_components=1)
                proj1 = pca.fit_transform(part1).ravel()
                proj2 = pca.fit_transform(part2).ravel()
                _, p_value = ttest_ind(proj1, proj2)

            significance_score = max(0, 1 - p_value)
        except:
            significance_score = 0.5

        total_score = (
                compactness_improvement * 0.3 +
                min(separation_score, 2.0) * 0.25 +
                balance_score * 0.2 +
                significance_score * 0.25
        )

        return max(0, total_score)

    def _should_split_cluster(self, cluster, iteration):
        """Рішення про поділ кластера"""
        cluster_data = cluster['data']
        cluster_size = len(cluster_data)

        if cluster_size < self.min_cluster_size or self.n_clusters >= self.max_clusters:
            return False, None, 0.0

        spread = self._calculate_cluster_spread(cluster_data)
        split_result, split_score = self._enhanced_split_analysis(cluster_data)

        if iteration <= 2:
            threshold = self.initial_split_threshold
        else:
            threshold = self.refinement_threshold

        size_factor = min(2.0, cluster_size / 50.0)
        final_score = split_score * size_factor
        should_split = final_score > threshold

        return should_split, split_result, final_score

    def _calculate_merge_score(self, cluster1, cluster2):
        """Розрахунок score для об'єднання двох кластерів"""
        data1, data2 = cluster1['data'], cluster2['data']

        # Відстань між центрами
        center_distance = np.linalg.norm(cluster1['mean'] - cluster2['mean'])

        # Розміри кластерів
        size1, size2 = len(data1), len(data2)

        # Симетрія розмірів (краще об'єднувати схожі за розміром)
        size_balance = min(size1, size2) / max(size1, size2)

        # Компактність після об'єднання
        merged_data = np.vstack([data1, data2])
        merged_center = np.mean(merged_data, axis=0)
        merged_compactness = np.mean(np.linalg.norm(merged_data - merged_center, axis=1))

        # Середня компактність до об'єднання
        comp1 = np.mean(np.linalg.norm(data1 - cluster1['mean'], axis=1))
        comp2 = np.mean(np.linalg.norm(data2 - cluster2['mean'], axis=1))
        avg_compactness = (comp1 * size1 + comp2 * size2) / (size1 + size2)

        # Збільшення компактності (негативне означає погіршення)
        compactness_change = (avg_compactness - merged_compactness) / avg_compactness

        # Нормована відстань
        avg_spread = (self._calculate_cluster_spread(data1) + self._calculate_cluster_spread(data2)) / 2
        normalized_distance = center_distance / (avg_spread + 1e-8)

        # Комбінований score (більший = краще для об'єднання)
        merge_score = (
                size_balance * 0.3 +
                compactness_change * 0.4 +
                (1 / (normalized_distance + 1)) * 0.3
        )

        return merge_score, normalized_distance

    def _backward_optimization(self, X):
        """Зворотний прохід для оптимізації кількості кластерів"""
        print(f"\n=== BACKWARD OPTIMIZATION ===")
        print(f"Starting with {self.n_clusters} clusters")

        # Зберігаємо поточну найкращу конфігурацію
        current_labels = self.predict(X)
        if len(np.unique(current_labels)) > 1:
            current_score = silhouette_score(X, current_labels)
            self.best_score = current_score
            self.best_configuration = {
                'clusters': [cluster.copy() for cluster in self.clusters],
                'n_clusters': self.n_clusters,
                'score': current_score,
                'labels': current_labels.copy()
            }
            print(f"Initial score: {current_score:.4f}")

        iteration = 0
        while self.n_clusters > 2:
            iteration += 1
            print(f"\n[BACKWARD {iteration}] Current clusters: {self.n_clusters}")

            # Знаходимо найкращу пару для об'єднання
            best_merge = None
            best_merge_score = -np.inf

            for i in range(self.n_clusters):
                for j in range(i + 1, self.n_clusters):
                    merge_score, distance = self._calculate_merge_score(self.clusters[i], self.clusters[j])

                    if merge_score > best_merge_score:
                        best_merge_score = merge_score
                        best_merge = (i, j, distance)

            if best_merge is None:
                break

            i, j, distance = best_merge
            print(f"  Merging clusters {i + 1} and {j + 1} (score: {best_merge_score:.3f}, dist: {distance:.3f})")

            # Виконуємо об'єднання
            cluster1 = self.clusters[i]
            cluster2 = self.clusters[j]

            merged_data = np.vstack([cluster1['data'], cluster2['data']])
            merged_cluster = {
                'data': merged_data,
                'mean': np.mean(merged_data, axis=0),
                'cov': np.cov(merged_data.T) + np.eye(merged_data.shape[1]) * 1e-6,
                'weight': cluster1['weight'] + cluster2['weight'],
                'size': len(merged_data),
                'id': cluster1['id']
            }

            # Оновлюємо список кластерів
            new_clusters = []
            for k, cluster in enumerate(self.clusters):
                if k not in [i, j]:
                    new_clusters.append(cluster)
            new_clusters.append(merged_cluster)

            self.clusters = new_clusters
            self.n_clusters = len(self.clusters)

            # Оцінюємо нову конфігурацію
            new_labels = self.predict(X)
            if len(np.unique(new_labels)) > 1:
                new_score = silhouette_score(X, new_labels)
                ari_score = adjusted_rand_score(y_true, new_labels) if 'y_true' in globals() else 0

                print(f"  New score: {new_score:.4f} (ARI: {ari_score:.4f})")

                # Зберігаємо якщо краще
                if new_score > self.best_score:
                    self.best_score = new_score
                    self.best_configuration = {
                        'clusters': [cluster.copy() for cluster in self.clusters],
                        'n_clusters': self.n_clusters,
                        'score': new_score,
                        'labels': new_labels.copy()
                    }
                    print(f"  ✅ New best configuration!")

                self.backward_history.append({
                    'n_clusters': self.n_clusters,
                    'silhouette': new_score,
                    'ari': ari_score,
                    'merge_score': best_merge_score
                })

            # Зупиняємося якщо досягли цільової кількості
            if self.target_clusters and self.n_clusters <= self.target_clusters:
                print(f"  Reached target clusters: {self.target_clusters}")
                break

        # Відновлюємо найкращу конфігурацію
        if self.best_configuration:
            print(f"\n[RESTORE] Restoring best configuration:")
            print(f"  Clusters: {self.best_configuration['n_clusters']}")
            print(f"  Score: {self.best_configuration['score']:.4f}")

            self.clusters = self.best_configuration['clusters']
            self.n_clusters = self.best_configuration['n_clusters']

    def predict(self, X):
        """Передбачення міток кластерів"""
        if self.n_clusters == 1:
            return np.zeros(len(X), dtype=int)

        labels = np.zeros(len(X))

        for i, point in enumerate(X):
            best_cluster = 0
            best_score = -np.inf

            for j, cluster in enumerate(self.clusters):
                try:
                    likelihood = multivariate_normal.logpdf(point, cluster['mean'], cluster['cov'])
                    distance = np.linalg.norm(point - cluster['mean'])
                    score = likelihood - distance * 0.1 + np.log(cluster['weight'])

                    if score > best_score:
                        best_score = score
                        best_cluster = j
                except:
                    distance = np.linalg.norm(point - cluster['mean'])
                    score = -distance + np.log(cluster['weight'])

                    if score > best_score:
                        best_score = score
                        best_cluster = j

            labels[i] = best_cluster

        return labels.astype(int)

    def fit(self, X):
        """Основний алгоритм навчання з зворотним проходом"""
        print("=== Improved Molecular DPMM with Backward Optimization ===")
        print(f"Alpha: {self.alpha}, Min size: {self.min_cluster_size}")
        print(f"Backward optimization: {self.backward_optimization}")

        # FORWARD PASS - Поділи кластерів
        self._initialize_clusters(X)
        print(f"[INIT] Starting with {self.n_clusters} cluster(s)")

        for iteration in range(self.max_iter):
            print(f"\n[FORWARD {iteration + 1}] Current clusters: {self.n_clusters}")

            new_clusters = []
            any_split = False

            for i, cluster in enumerate(self.clusters):
                cluster_size = cluster['size']
                print(f"  Cluster {i + 1}: size={cluster_size}", end='')

                should_split, split_result, score = self._should_split_cluster(cluster, iteration)
                print(f", score={score:.3f}", end='')

                if should_split and split_result is not None:
                    part1, part2 = split_result
                    print(f" → SPLIT ✅ ({len(part1)}+{len(part2)})")

                    for part in [part1, part2]:
                        new_cluster = {
                            'data': part,
                            'mean': np.mean(part, axis=0),
                            'cov': np.cov(part.T) + np.eye(part.shape[1]) * 1e-6,
                            'weight': len(part) / len(X),
                            'size': len(part),
                            'id': len(new_clusters)
                        }
                        new_clusters.append(new_cluster)

                    any_split = True
                else:
                    print(" → keep")
                    new_clusters.append(cluster)

            self.clusters = new_clusters
            self.n_clusters = len(self.clusters)

            # Оцінка якості
            if self.n_clusters > 1:
                labels = self.predict(X)
                if len(np.unique(labels)) > 1:
                    quality = silhouette_score(X, labels)
                    self.quality_history.append(quality)
                    print(f"  Quality (Silhouette): {quality:.4f}")

            if not any_split:
                print("\n[FORWARD CONVERGENCE] No beneficial splits found")
                break

        # BACKWARD PASS - Оптимізація через об'єднання
        if self.backward_optimization and self.n_clusters > 2:
            self._backward_optimization(X)

        # Фінальна оцінка
        final_labels = self.predict(X)
        final_quality = silhouette_score(X, final_labels) if len(np.unique(final_labels)) > 1 else 0
        final_ari = adjusted_rand_score(y_true, final_labels) if 'y_true' in globals() else 0

        print(f"\n[FINAL] Clusters: {self.n_clusters}")
        print(f"        Silhouette: {final_quality:.4f}")
        print(f"        ARI: {final_ari:.4f}")

        return self


# === Тестування з зворотним проходом ===

models = [
    ("Standard", ImprovedMolecularDPMM(
        alpha=1.0, min_cluster_size=15,
        initial_split_threshold=0.25, refinement_threshold=0.45,
        max_clusters=12, use_smart_init=True,
        backward_optimization=False
    )),
    ("With Backward", ImprovedMolecularDPMM(
        alpha=1.0, min_cluster_size=15,
        initial_split_threshold=0.25, refinement_threshold=0.45,
        max_clusters=12, use_smart_init=True,
        backward_optimization=True
    )),
    ("Target-Aware", ImprovedMolecularDPMM(
        alpha=1.0, min_cluster_size=15,
        initial_split_threshold=0.2, refinement_threshold=0.4,
        max_clusters=15, use_smart_init=True,
        backward_optimization=True, target_clusters=4
    ))
]

results = []

for name, model in models:
    print(f"\n{'=' * 80}")
    print(f"Testing: {name}")
    print('=' * 80)

    model.fit(X)
    labels = model.predict(X)

    n_clusters = len(np.unique(labels))
    if n_clusters > 1:
        sil_score = silhouette_score(X, labels)
        ari_score = adjusted_rand_score(y_true, labels)
        ch_score = calinski_harabasz_score(X, labels)
    else:
        sil_score = ari_score = ch_score = 0

    results.append({
        'name': name,
        'model': model,
        'labels': labels,
        'n_clusters': n_clusters,
        'silhouette': sil_score,
        'ari': ari_score,
        'calinski_harabasz': ch_score,
        'backward_history': getattr(model, 'backward_history', [])
    })

# === Візуалізація ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Оригінальні дані
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=30, alpha=0.8)
axes[0].set_title(f'True Clusters (n={len(np.unique(y_true))})', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Результати
for i, result in enumerate(results):
    ax = axes[i + 1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                         cmap='tab20', s=30, alpha=0.8)
    ax.set_title(f"{result['name']}\n"
                 f"n={result['n_clusters']}, Sil={result['silhouette']:.3f}, "
                 f"ARI={result['ari']:.3f}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Візуалізація історії зворотного проходу ===
backward_models = [r for r in results if r['backward_history']]

if backward_models:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for result in backward_models:
        history = result['backward_history']
        if history:
            n_clusters = [h['n_clusters'] for h in history]
            silhouettes = [h['silhouette'] for h in history]
            aris = [h['ari'] for h in history]

            axes[0].plot(n_clusters, silhouettes, 'o-', label=result['name'], linewidth=2, markersize=6)
            axes[1].plot(n_clusters, aris, 's-', label=result['name'], linewidth=2, markersize=6)

    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Backward Optimization: Silhouette Score')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('ARI Score')
    axes[1].set_title('Backward Optimization: ARI Score')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# === Результати ===
print(f"\n{'=' * 90}")
print("FINAL RESULTS WITH BACKWARD OPTIMIZATION")
print('=' * 90)
print(f"{'Model':<15} {'Clusters':<8} {'Silhouette':<11} {'ARI':<8} {'CH Score':<10} {'Best':<6}")
print('-' * 90)

best_ari = max(r['ari'] for r in results)
for result in results:
    is_best = "✅" if abs(result['ari'] - best_ari) < 0.001 else ""
    print(f"{result['name']:<15} {result['n_clusters']:<8} "
          f"{result['silhouette']:<11.4f} {result['ari']:<8.4f} "
          f"{result['calinski_harabasz']:<10.0f} {is_best:<6}")

print(f"\nTrue clusters: {len(np.unique(y_true))}")

# Показуємо деталі зворотного проходу
for result in results:
    if result['backward_history']:
        print(f"\n{result['name']} - Backward Optimization History:")
        print("Clusters → Silhouette | ARI")
        for h in result['backward_history']:
            print(f"   {h['n_clusters']:2d}    →   {h['silhouette']:.4f}  | {h['ari']:.4f}")