import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
import time
from scipy.stats import beta, gamma
import logging
import concurrent.futures

warnings.filterwarnings('ignore')

# Generate sample data
np.random.seed(42)
X, y_true = make_blobs(n_samples=6000, centers=10, cluster_std=0.6, random_state=42)


# === Додаткові DPMM моделі ===

class StandardDPMM:
    """Стандартна DPMM з простим поділом"""

    def __init__(self, alpha=1.0, max_iter=10, min_cluster_size=5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_cluster_size = min_cluster_size
        self.clusters = []
        self.n_clusters = 0

    def _initialize_single_cluster(self, X):
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

    def _simple_split(self, cluster_data):
        """Простий поділ по медіані найбільшої компоненти"""
        if len(cluster_data) < self.min_cluster_size:
            return None

        # Знаходимо напрямок з найбільшою варіацією
        variances = np.var(cluster_data, axis=0)
        split_dim = np.argmax(variances)

        # Ділимо по медіані
        median_val = np.median(cluster_data[:, split_dim])
        mask1 = cluster_data[:, split_dim] <= median_val
        mask2 = cluster_data[:, split_dim] > median_val

        if np.sum(mask1) >= 3 and np.sum(mask2) >= 3:
            return cluster_data[mask1], cluster_data[mask2]
        return None

    def predict(self, X):
        if self.n_clusters <= 1:
            return np.zeros(len(X), dtype=int)

        labels = np.zeros(len(X))
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - cluster['mean'])
                         for cluster in self.clusters]
            labels[i] = np.argmin(distances)
        return labels.astype(int)

    def fit(self, X):
        self._initialize_single_cluster(X)

        for iteration in range(self.max_iter):
            new_clusters = []
            any_split = False

            for cluster in self.clusters:
                if cluster['size'] >= self.min_cluster_size * 2:
                    split_result = self._simple_split(cluster['data'])
                    if split_result is not None:
                        part1, part2 = split_result
                        # Створюємо два нові кластери
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
                        new_clusters.append(cluster)
                else:
                    new_clusters.append(cluster)

            self.clusters = new_clusters
            self.n_clusters = len(self.clusters)

            if not any_split:
                break

        return self


class VariationalDPMM:
    """Варіаційна DPMM з використанням Bayesian Gaussian Mixture"""

    def __init__(self, max_components=20, alpha=1.0):
        self.max_components = max_components
        self.alpha = alpha
        self.model = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=alpha,
            random_state=42,
            max_iter=200,
            n_init=3
        )
        self.n_clusters = 0

    def fit(self, X):
        self.model.fit(X)
        # Визначаємо активні компоненти
        weights = self.model.weights_
        active_components = weights > 0.01  # Поріг для активних компонент
        self.n_clusters = np.sum(active_components)
        return self

    def predict(self, X):
        labels = self.model.predict(X)
        # Перемаппінг міток для активних компонент
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])


class HierarchicalDPMM:
    """Ієрархічна DPMM з поетапним поділом"""

    def __init__(self, alpha=1.0, max_depth=4, min_cluster_size=10):
        self.alpha = alpha
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.clusters = []
        self.n_clusters = 0

    def _split_cluster_hierarchical(self, data, depth=0):
        """Рекурсивний ієрархічний поділ"""
        if len(data) < self.min_cluster_size or depth >= self.max_depth:
            return [data]

        # Використовуємо K-means для поділу на 2
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            part1 = data[labels == 0]
            part2 = data[labels == 1]

            if len(part1) < 3 or len(part2) < 3:
                return [data]

            # Оцінюємо якість поділу
            silhouette = silhouette_score(data, labels)
            if silhouette < 0.3:  # Поріг якості
                return [data]

            # Рекурсивно ділимо кожну частину
            result = []
            result.extend(self._split_cluster_hierarchical(part1, depth + 1))
            result.extend(self._split_cluster_hierarchical(part2, depth + 1))
            return result

        except:
            return [data]

    def fit(self, X):
        # Ієрархічний поділ
        cluster_parts = self._split_cluster_hierarchical(X)

        # Створюємо кластери
        self.clusters = []
        for i, part in enumerate(cluster_parts):
            cluster = {
                'data': part,
                'mean': np.mean(part, axis=0),
                'cov': np.cov(part.T) + np.eye(part.shape[1]) * 1e-6,
                'weight': len(part) / len(X),
                'size': len(part),
                'id': i
            }
            self.clusters.append(cluster)

        self.n_clusters = len(self.clusters)
        return self

    def predict(self, X):
        if self.n_clusters <= 1:
            return np.zeros(len(X), dtype=int)

        labels = np.zeros(len(X))
        for i, point in enumerate(X):
            distances = [np.linalg.norm(point - cluster['mean'])
                         for cluster in self.clusters]
            labels[i] = np.argmin(distances)
        return labels.astype(int)


class AdaptiveDPMM:
    """Адаптивна DPMM з динамічними параметрами"""

    def __init__(self, initial_alpha=1.0, min_cluster_size=5):
        self.initial_alpha = initial_alpha
        self.min_cluster_size = min_cluster_size
        self.clusters = []
        self.n_clusters = 0

    def _adaptive_split_threshold(self, cluster_size, iteration):
        """Адаптивний поріг поділу"""
        base_threshold = 0.3
        size_factor = min(2.0, cluster_size / 50.0)
        iteration_decay = 0.9 ** iteration
        return base_threshold * size_factor * iteration_decay

    def _evaluate_split_adaptive(self, data):
        """Адаптивна оцінка поділу"""
        if len(data) < 6:
            return None, 0.0

        best_split = None
        best_score = 0.0

        # Пробуємо різні методи поділу
        methods = ['kmeans', 'pca', 'variance']

        for method in methods:
            if method == 'kmeans':
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
                    labels = kmeans.fit_predict(data)
                    part1, part2 = data[labels == 0], data[labels == 1]
                except:
                    continue

            elif method == 'pca':
                try:
                    pca = PCA(n_components=1)
                    projections = pca.fit_transform(data).ravel()
                    median_val = np.median(projections)
                    mask1 = projections <= median_val
                    mask2 = projections > median_val
                    part1, part2 = data[mask1], data[mask2]
                except:
                    continue

            elif method == 'variance':
                variances = np.var(data, axis=0)
                split_dim = np.argmax(variances)
                median_val = np.median(data[:, split_dim])
                mask1 = data[:, split_dim] <= median_val
                mask2 = data[:, split_dim] > median_val
                part1, part2 = data[mask1], data[mask2]

            if len(part1) >= 3 and len(part2) >= 3:
                # Оцінюємо якість поділу
                try:
                    combined_data = np.vstack([part1, part2])
                    labels = np.hstack([np.zeros(len(part1)), np.ones(len(part2))])
                    score = silhouette_score(combined_data, labels)

                    if score > best_score:
                        best_score = score
                        best_split = (part1, part2)
                except:
                    continue

        return best_split, best_score

    def fit(self, X):
        # Початкова ініціалізація
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

        for iteration in range(8):
            new_clusters = []
            any_split = False

            for cluster in self.clusters:
                cluster_size = cluster['size']
                threshold = self._adaptive_split_threshold(cluster_size, iteration)

                if cluster_size >= self.min_cluster_size * 2:
                    split_result, score = self._evaluate_split_adaptive(cluster['data'])

                    if split_result is not None and score > threshold:
                        part1, part2 = split_result

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
                        new_clusters.append(cluster)
                else:
                    new_clusters.append(cluster)

            self.clusters = new_clusters
            self.n_clusters = len(self.clusters)

            if not any_split:
                break

        return self

    def predict(self, X):
        if self.n_clusters <= 1:
            return np.zeros(len(X), dtype=int)

        labels = np.zeros(len(X))
        for i, point in enumerate(X):
            best_cluster = 0
            best_score = -np.inf

            for j, cluster in enumerate(self.clusters):
                try:
                    likelihood = multivariate_normal.logpdf(point, cluster['mean'], cluster['cov'])
                    score = likelihood + np.log(cluster['weight'])
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


import numpy as np
from scipy.stats import invwishart, multivariate_normal
from sklearn.base import BaseEstimator, ClusterMixin


class ImprovedMolecularDPMM(BaseEstimator, ClusterMixin):
    """
    DP-мікс Гаусіан з умовно-кон'югатними NIW-priors та Gibbs sampling.
    Реалізує sklearn-совісний API: fit, predict, predict_proba, score, log_likelihood.
    """

    def __init__(self, alpha=1.0, n_iter=100, kappa_0=None, xi_0=None, beta_0=None, W_0=None, verbose=False):
        """
        alpha: концентраційний параметр DP.
        n_iter: кількість ітерацій Gibbs sampling.
        kappa_0, xi_0, beta_0, W_0: гіперпараметри NIW prior (за замовчуванням задаються по даним).
        """
        self.alpha = alpha
        self.n_iter = n_iter
        self.kappa_0 = kappa_0
        self.xi_0 = xi_0
        self.beta_0 = beta_0
        self.W_0 = W_0
        self.verbose = verbose

    def _initialize_hyperparams(self, X):
        """Ініціалізація гіперпараметрів на основі даних X, якщо вони не задані."""
        N, D = X.shape
        sample_mean = np.mean(X, axis=0)
        sample_cov = np.cov(X.T) + np.eye(D) * 1e-6
        # Якщо параметри не задані, ставимо 'невілюючі' значення за емпіричними оцінками
        if self.xi_0 is None:
            self.xi_0 = sample_mean
        if self.kappa_0 is None:
            self.kappa_0 = 1.0
        if self.beta_0 is None:
            self.beta_0 = D + 2
        if self.W_0 is None:
            self.W_0 = sample_cov

    def _sample_cluster_params(self, Xj):
        """
        Зразкування параметрів кластера (mu, Sigma) з posterior NIW,
        де Xj — дані кластера. Якщо даних немає, використовується prior.
        """
        D = Xj.shape[1]
        n_j = Xj.shape[0]
        if n_j > 0:
            # Оновлюємо NIW-параметри
            x_mean = np.mean(Xj, axis=0)
            S = np.dot((Xj - x_mean).T, (Xj - x_mean))
            kappa_n = self.kappa_0 + n_j
            xi_n = (self.kappa_0 * self.xi_0 + n_j * x_mean) / kappa_n
            beta_n = self.beta_0 + n_j
            diff = x_mean - self.xi_0
            W_n = self.W_0 + S + (self.kappa_0 * n_j / kappa_n) * np.outer(diff, diff)
            # Зразок Sigma ~ InvWishart(beta_n, W_n) та mu ~ N(xi_n, Sigma/kappa_n)
            Sigma = invwishart.rvs(df=beta_n, scale=W_n)
            mu = multivariate_normal.rvs(mean=xi_n, cov=Sigma / kappa_n)
            return mu, Sigma
        else:
            # Немає даних: зразкуємо з prior
            Sigma = invwishart.rvs(df=self.beta_0, scale=self.W_0)
            mu = multivariate_normal.rvs(mean=self.xi_0, cov=Sigma / self.kappa_0)
            return mu, Sigma

    def fit(self, X, y=None):
        """
        Навчання моделі DP-GMM за допомогою Gibbs sampling з безпечною переіндексацією кластерів.
        """
        X = np.asarray(X)
        N, D = X.shape
        self._initialize_hyperparams(X)

        # Ініціалізація: усі точки в одному кластері
        assignments = np.zeros(N, dtype=int)
        clusters = {0: list(range(N))}
        mu_list = []
        Sigma_list = []

        mu_0, Sigma_0 = self._sample_cluster_params(X)
        mu_list.append(mu_0)
        Sigma_list.append(Sigma_0)

        for it in range(self.n_iter):
            if self.verbose:
                print(f"Iteration {it + 1}/{self.n_iter}")
                print(f"  Clusters: {len(clusters)}")

            for i in range(N):
                curr_cluster = assignments[i]
                # Видаляємо точку i з поточного кластеру
                clusters[curr_cluster].remove(i)

                # Якщо кластер порожній — видаляємо його
                if len(clusters[curr_cluster]) == 0:
                    del clusters[curr_cluster]

                    # Створюємо нову переіндексацію
                    old_to_new = {}
                    new_clusters = {}
                    new_mu_list = []
                    new_Sigma_list = []
                    new_idx = 0

                    for old_idx in sorted(clusters.keys()):
                        old_to_new[old_idx] = new_idx
                        new_clusters[new_idx] = clusters[old_idx]
                        new_mu_list.append(mu_list[old_idx])
                        new_Sigma_list.append(Sigma_list[old_idx])
                        new_idx += 1

                    clusters = new_clusters
                    mu_list = new_mu_list
                    Sigma_list = new_Sigma_list

                    # Оновлюємо assignments
                    for j in range(N):
                        if assignments[j] == curr_cluster:
                            assignments[j] = -1  # потрібно перепризначити
                        elif assignments[j] > curr_cluster:
                            assignments[j] -= 1

                    # Оновлюємо поточний кластер
                    curr_cluster = -1

                # Обчислюємо лог-імовірності для існуючих кластерів
                K = len(mu_list)
                log_probs = []

                for j in range(K):
                    nj = len(clusters[j])
                    ll = multivariate_normal.logpdf(X[i], mean=mu_list[j],
                                                    cov=Sigma_list[j] + 1e-6 * np.eye(D))
                    log_probs.append(np.log(nj) + ll)

                # Додаємо ймовірність нового кластеру
                base_cov = self.W_0 * (self.kappa_0 + 1) / (self.kappa_0 * (self.beta_0 - D + 1))
                ll_base = multivariate_normal.logpdf(X[i], mean=self.xi_0, cov=base_cov + 1e-6 * np.eye(D))
                log_probs.append(np.log(self.alpha) + ll_base)

                # Вибір нового кластера
                max_lp = max(log_probs)
                probs = np.exp(np.array(log_probs) - max_lp)
                probs /= probs.sum()
                choice = np.random.choice(len(probs), p=probs)

                if choice < K:
                    assignments[i] = choice
                    clusters[choice].append(i)
                else:
                    new_idx = K
                    assignments[i] = new_idx
                    clusters[new_idx] = [i]
                    mu_new, Sigma_new = self._sample_cluster_params(X[[i]])
                    mu_list.append(mu_new)
                    Sigma_list.append(Sigma_new)

            # Переоцінка параметрів кластерів
            for j, indices in clusters.items():
                Xj = X[indices] if indices else np.empty((0, D))
                mu_j, Sigma_j = self._sample_cluster_params(Xj)
                mu_list[j] = mu_j
                Sigma_list[j] = Sigma_j

        # Зберігаємо параметри
        self.cluster_centers_ = np.array(mu_list)
        self.covariances_ = Sigma_list
        self.assignments_ = assignments
        self.n_components_ = len(mu_list)
        weights = np.array([len(clusters[j]) for j in range(self.n_components_)])
        self.weights_ = weights / weights.sum()

        return self

    def predict(self, X):
        """
        Присвоєння кластера кожній точці: максимізуємо log p(x|µ_j,Σ_j) + log w_j.
        """
        X = np.asarray(X)
        N, D = X.shape
        labels = np.zeros(N, dtype=int)
        for i in range(N):
            log_ps = []
            for j in range(self.n_components_):
                ll = multivariate_normal.logpdf(X[i], mean=self.cluster_centers_[j],
                                                cov=self.covariances_[j] + 1e-6 * np.eye(D))
                log_ps.append(ll + np.log(self.weights_[j] + 1e-16))
            labels[i] = np.argmax(log_ps)
        return labels

    def predict_proba(self, X):
        """
        Ймовірністі приналежності до кластерів (normalized softmax від log-потенціалів).
        """
        X = np.asarray(X)
        N, D = X.shape
        log_probs = np.zeros((N, self.n_components_))
        for j in range(self.n_components_):
            ll = multivariate_normal.logpdf(X, mean=self.cluster_centers_[j],
                                            cov=self.covariances_[j] + 1e-6 * np.eye(D))
            log_probs[:, j] = ll + np.log(self.weights_[j] + 1e-16)
        # softmax normalisation
        max_ll = np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - max_ll)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def log_likelihood(self, X):
        """
        Сума log-щільностей змішаної моделі: log ∏_i ∑_j w_j N(x_i|µ_j,Σ_j).
        """
        X = np.asarray(X)
        N, D = X.shape
        logL = 0.0
        for i in range(N):
            mix_val = 0.0
            for j in range(self.n_components_):
                mix_val += self.weights_[j] * multivariate_normal.pdf(
                    X[i], mean=self.cluster_centers_[j],
                    cov=self.covariances_[j] + 1e-6 * np.eye(D))
            logL += np.log(mix_val + 1e-16)
        return logL

    def leave_one_out_log_likelihood(self, X):
        """
        Leave-one-out predictive log-likelihood: для кожної точки i обчислює
        log p(x_i | X_{\setminus i}). Сумуємо їх.
        """
        X = np.asarray(X)
        N, D = X.shape
        loo_logL = 0.0
        counts = np.bincount(self.assignments_, minlength=self.n_components_)
        for i in range(N):
            k = self.assignments_[i]
            n_k = counts[k] - 1
            total = N - 1 + self.alpha
            p_i = 0.0
            for j in range(self.n_components_):
                n_j = (counts[j] if j != k else n_k)
                if n_j > 0:
                    p_ij = (n_j / total) * multivariate_normal.pdf(
                        X[i], mean=self.cluster_centers_[j],
                        cov=self.covariances_[j] + 1e-6 * np.eye(D))
                    p_i += p_ij
            # Новий кластер
            base_cov = self.W_0 * (self.kappa_0 + 1) / (self.kappa_0 * (self.beta_0 - D + 1))
            p_new = (self.alpha / total) * multivariate_normal.pdf(
                X[i], mean=self.xi_0, cov=base_cov + 1e-6 * np.eye(D))
            p_i += p_new
            loo_logL += np.log(p_i + 1e-16)
        return loo_logL

    def score(self, X):
        """
        sklearn-ковський score: повертає середній LOO predictive log-лікенс.
        """
        X = np.asarray(X)
        loo = self.leave_one_out_log_likelihood(X)
        return loo / X.shape[0]


# === Комплексне тестування ===

def evaluate_model(model, X, y_true, name):
    """Оцінка моделі з метриками часу"""
    start_time = time.time()

    try:
        model.fit(X)
        labels = model.predict(X)

        fit_time = time.time() - start_time

        n_clusters = len(np.unique(labels))

        if n_clusters > 1:
            sil_score = silhouette_score(X, labels)
            ari_score = adjusted_rand_score(y_true, labels)
            ch_score = calinski_harabasz_score(X, labels)
        else:
            sil_score = ari_score = ch_score = 0

        return {
            'name': name,
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette': sil_score,
            'ari': ari_score,
            'calinski_harabasz': ch_score,
            'fit_time': fit_time,
            'success': True
        }

    except Exception as e:
        return {
            'name': name,
            'model': model,
            'labels': np.zeros(len(X)),
            'n_clusters': 1,
            'silhouette': 0,
            'ari': 0,
            'calinski_harabasz': 0,
            'fit_time': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


# Модель для порівняння
models_to_compare = [
    # Базові методи
    ("Standard DPMM", StandardDPMM(alpha=1.0, min_cluster_size=5)),
    ("Variational DPMM", VariationalDPMM(max_components=15, alpha=1.0)),
    ("Hierarchical DPMM", HierarchicalDPMM(alpha=1.0, max_depth=4, min_cluster_size=5)),
    ("Adaptive DPMM", AdaptiveDPMM(initial_alpha=1.0, min_cluster_size=5)),

    # Нові покращені версії з Bayesian inference
    ("Improved Standard", ImprovedMolecularDPMM(
        alpha=1.0,
        n_iter=100,
        verbose=True
    )),
    ("Improved + Backward", ImprovedMolecularDPMM(
        alpha=1.0,
        n_iter=100,
        verbose=True  # наприклад, для відображення прогресу
    )),

    # Еталонні методи для порівняння
    ("Gaussian Mixture", GaussianMixture(n_components=6, random_state=42)),
    ("Bayesian GMM", BayesianGaussianMixture(n_components=15, random_state=42)),
    ("K-Means", KMeans(n_clusters=6, random_state=42))
]

print("=" * 80)
print("COMPREHENSIVE DPMM COMPARISON ANALYSIS")
print("=" * 80)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y_true))} true clusters")
print()

results = []
for name, model in models_to_compare:
    print(f"Testing {name}...", end=" ")
    result = evaluate_model(model, X, y_true, name)
    results.append(result)

    if result['success']:
        print(f"✅ ({result['fit_time']:.2f}s, {result['n_clusters']} clusters)")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown')}")

# === Результати ===
print(f"\n{'=' * 100}")
print("DETAILED COMPARISON RESULTS")
print('=' * 100)
print(f"{'Model':<20} {'Clusters':<8} {'Silhouette':<11} {'ARI':<8} {'CH Score':<10} {'Time(s)':<8} {'Status'}")
print('-' * 100)

successful_results = [r for r in results if r['success']]
best_ari = max(r['ari'] for r in successful_results) if successful_results else 0

for result in results:
    if result['success']:
        is_best = "🏆" if abs(result['ari'] - best_ari) < 0.001 and result['ari'] > 0 else ""
        status = f"✅ {is_best}"
    else:
        status = "❌"

    print(f"{result['name']:<20} {result['n_clusters']:<8} "
          f"{result['silhouette']:<11.4f} {result['ari']:<8.4f} "
          f"{result['calinski_harabasz']:<10.0f} "
          f"{result['fit_time']:<8.2f} {status}")

print(f"\nTrue clusters: {len(np.unique(y_true))}")

# === Статистичний аналіз ===
print(f"\n{'=' * 60}")
print("STATISTICAL ANALYSIS")
print('=' * 60)

if successful_results:
    # Групування по типам
    dpmm_results = [r for r in successful_results if 'DPMM' in r['name']]
    improved_results = [r for r in successful_results if 'Improved' in r['name']]
    baseline_results = [r for r in successful_results if r['name'] in ['Gaussian Mixture', 'Bayesian GMM', 'K-Means']]


    def print_group_stats(group, group_name):
        if not group:
            return
        aris = [r['ari'] for r in group]
        silhouettes = [r['silhouette'] for r in group]
        times = [r['fit_time'] for r in group]

        print(f"\n{group_name}:")
        print(f"  ARI: mean={np.mean(aris):.4f}, std={np.std(aris):.4f}, max={np.max(aris):.4f}")
        print(f"  Silhouette: mean={np.mean(silhouettes):.4f}, std={np.std(silhouettes):.4f}")
        print(f"  Time: mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")


    print_group_stats(dpmm_results, "DPMM Methods")
    print_group_stats(improved_results, "Improved Methods")
    print_group_stats(baseline_results, "Baseline Methods")

# === Візуалізація ===
fig = plt.figure(figsize=(20, 16))

# Створюємо сітку для підплотів
n_models = len(successful_results) + 1  # +1 для true labels
n_cols = 4
n_rows = (n_models + n_cols - 1) // n_cols

# True clusters
ax = plt.subplot(n_rows, n_cols, 1)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=15, alpha=0.7)
ax.set_title(f'True Clusters (n={len(np.unique(y_true))})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Результати моделей
for i, result in enumerate(successful_results, 2):
    ax = plt.subplot(n_rows, n_cols, i)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=result['labels'],
                         cmap='tab20', s=15, alpha=0.7)

    title = f"{result['name']}\n"
    title += f"n={result['n_clusters']}, ARI={result['ari']:.3f}\n"
    title += f"Sil={result['silhouette']:.3f}, t={result['fit_time']:.1f}s"

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === Метрики порівняння ===
if len(successful_results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ARI vs Time
    aris = [r['ari'] for r in successful_results]
    times = [r['fit_time'] for r in successful_results]
    names = [r['name'] for r in successful_results]

    axes[0, 0].scatter(times, aris, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 0].annotate(name, (times[i], aris[i]), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)
    axes[0, 0].set_xlabel('Fit Time (seconds)')
    axes[0, 0].set_ylabel('ARI Score')
    axes[0, 0].set_title('ARI vs Computation Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Silhouette vs Clusters
    silhouettes = [r['silhouette'] for r in successful_results]
    n_clusters = [r['n_clusters'] for r in successful_results]

    axes[0, 1].scatter(n_clusters, silhouettes, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 1].annotate(name, (n_clusters[i], silhouettes[i]), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)
    axes[0, 1].axvline(x=len(np.unique(y_true)), color='red', linestyle='--', alpha=0.7, label='True clusters')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette vs Number of Clusters')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ARI comparison bar chart
    sorted_results = sorted(successful_results, key=lambda x: x['ari'], reverse=True)
    names_sorted = [r['name'] for r in sorted_results]
    aris_sorted = [r['ari'] for r in sorted_results]

    bars = axes[1, 0].bar(range(len(names_sorted)), aris_sorted, alpha=0.7)
    axes[1, 0].set_xticks(range(len(names_sorted)))
    axes[1, 0].set_xticklabels(names_sorted, rotation=45, ha='right')
    axes[1, 0].set_ylabel('ARI Score')
    axes[1, 0].set_title('ARI Score Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Добавляем цветовое кодирование
    for i, bar in enumerate(bars):
        if i == 0:  # Лучший результат
            bar.set_color('gold')
        elif 'Improved' in names_sorted[i]:
            bar.set_color('lightgreen')
        elif 'DPMM' in names_sorted[i]:
            bar.set_color('lightblue')
        else:
            bar.set_color('lightcoral')

    # Performance matrix (ARI vs Silhouette)
    colors = []
    for result in successful_results:
        if 'Improved' in result['name']:
            colors.append('green')
        elif 'DPMM' in result['name']:
            colors.append('blue')
        else:
            colors.append('red')

    scatter = axes[1, 1].scatter(silhouettes, aris, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[1, 1].annotate(name, (silhouettes[i], aris[i]), xytext=(5, 5),
                            textcoords='offset points', fontsize=9)
    axes[1, 1].set_xlabel('Silhouette Score')
    axes[1, 1].set_ylabel('ARI Score')
    axes[1, 1].set_title('Performance Matrix (ARI vs Silhouette)')
    axes[1, 1].grid(True, alpha=0.3)

    # Добавляем легенду для цветов
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor='green', label='Improved Methods'),
                       Patch(facecolor='blue', label='DPMM Methods'),
                       Patch(facecolor='red', label='Baseline Methods')]
    axes[1, 1].legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()

# === Детальний аналіз найкращих методів ===
print(f"\n{'=' * 80}")
print("DETAILED ANALYSIS OF TOP PERFORMERS")
print('=' * 80)

# Топ-3 по ARI
top_3_ari = sorted(successful_results, key=lambda x: x['ari'], reverse=True)[:3]
print("Top 3 by ARI Score:")
for i, result in enumerate(top_3_ari, 1):
    print(f"{i}. {result['name']}: ARI={result['ari']:.4f}, "
          f"Silhouette={result['silhouette']:.4f}, "
          f"Clusters={result['n_clusters']}, Time={result['fit_time']:.2f}s")

# Топ-3 по Silhouette
top_3_sil = sorted(successful_results, key=lambda x: x['silhouette'], reverse=True)[:3]
print(f"\nTop 3 by Silhouette Score:")
for i, result in enumerate(top_3_sil, 1):
    print(f"{i}. {result['name']}: Silhouette={result['silhouette']:.4f}, "
          f"ARI={result['ari']:.4f}, "
          f"Clusters={result['n_clusters']}, Time={result['fit_time']:.2f}s")

# Найшвидші методи
fastest_3 = sorted(successful_results, key=lambda x: x['fit_time'])[:3]
print(f"\nTop 3 by Speed:")
for i, result in enumerate(fastest_3, 1):
    print(f"{i}. {result['name']}: Time={result['fit_time']:.2f}s, "
          f"ARI={result['ari']:.4f}, "
          f"Silhouette={result['silhouette']:.4f}")

# Збалансований рейтинг
print(f"\n{'=' * 60}")
print("BALANCED RANKING (ARI + Silhouette + Speed)")
print('=' * 60)

# Нормалізуємо метрики
max_ari = max(r['ari'] for r in successful_results)
max_sil = max(r['silhouette'] for r in successful_results)
min_time = min(r['fit_time'] for r in successful_results)

balanced_scores = []
for result in successful_results:
    # Нормалізовані метрики (0-1)
    norm_ari = result['ari'] / max_ari if max_ari > 0 else 0
    norm_sil = result['silhouette'] / max_sil if max_sil > 0 else 0
    norm_speed = min_time / result['fit_time'] if result['fit_time'] > 0 else 0

    # Збалансований скор (вага: ARI=40%, Silhouette=40%, Speed=20%)
    balanced_score = norm_ari * 0.4 + norm_sil * 0.4 + norm_speed * 0.2

    balanced_scores.append({
        'name': result['name'],
        'score': balanced_score,
        'ari': result['ari'],
        'silhouette': result['silhouette'],
        'time': result['fit_time'],
        'clusters': result['n_clusters']
    })

# Сортуємо по збалансованому скору
balanced_scores.sort(key=lambda x: x['score'], reverse=True)

print(f"{'Rank':<4} {'Model':<20} {'Score':<7} {'ARI':<7} {'Sil':<7} {'Time':<7} {'Clusters'}")
print('-' * 70)
for i, result in enumerate(balanced_scores, 1):
    print(f"{i:<4} {result['name']:<20} {result['score']:.3f}   "
          f"{result['ari']:.3f}   {result['silhouette']:.3f}   "
          f"{result['time']:.2f}s   {result['clusters']}")

# === Рекомендації ===
print(f"\n{'=' * 80}")
print("RECOMMENDATIONS")
print('=' * 80)

best_overall = balanced_scores[0]
best_ari = max(successful_results, key=lambda x: x['ari'])
fastest = min(successful_results, key=lambda x: x['fit_time'])

print(f"🏆 BEST OVERALL: {best_overall['name']}")
print(f"   → Balanced performance across all metrics")
print(f"   → Score: {best_overall['score']:.3f}, ARI: {best_overall['ari']:.3f}")

print(f"\n🎯 HIGHEST ACCURACY: {best_ari['name']}")
print(f"   → Best cluster recovery (ARI: {best_ari['ari']:.3f})")
print(f"   → Use when accuracy is most important")

print(f"\n⚡ FASTEST: {fastest['name']}")
print(f"   → Fastest execution ({fastest['fit_time']:.2f}s)")
print(f"   → Use for large datasets or real-time applications")

# Специфічні рекомендації
print(f"\n📋 SPECIFIC USE CASES:")
print(f"   • High-dimensional data: {max(successful_results, key=lambda x: x['silhouette'])['name']}")
print(
    f"   • Unknown cluster count: {[r['name'] for r in successful_results if 'Improved' in r['name']][0] if any('Improved' in r['name'] for r in successful_results) else 'N/A'}")
print(f"   • Real-time processing: {fastest['name']}")
print(f"   • Research/exploration: {best_ari['name']}")

# === Підсумок ===
print(f"\n{'=' * 80}")
print("SUMMARY")
print('=' * 80)
print(f"• Total models tested: {len(models_to_compare)}")
print(f"• Successful runs: {len(successful_results)}")
print(f"• True clusters: {len(np.unique(y_true))}")
print(f"• Best ARI achieved: {best_ari['ari']:.4f} ({best_ari['name']})")
print(f"• Best Silhouette: {max(r['silhouette'] for r in successful_results):.4f}")
print(
    f"• Execution time range: {min(r['fit_time'] for r in successful_results):.2f}s - {max(r['fit_time'] for r in successful_results):.2f}s")

print(
    f"\n✅ Analysis complete! The improved DPMM methods show {'superior' if any('Improved' in r['name'] for r in balanced_scores[:3]) else 'competitive'} performance.")

# === Експорт результатів ===
print(f"\n📊 Results saved to variables:")
print(f"   • results: List of all model results")
print(f"   • balanced_scores: Ranked by balanced performance")
print(f"   • successful_results: Only successful model runs")
print(f"   • X, y_true: Original dataset and true labels")