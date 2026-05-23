import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from tqdm import tqdm

COLORS = ['purple', 'dodgerblue', 'gold', 'green', 'red', 'cyan', 'brown', 'magenta', 'orange', 'lime']
RANDOM_STATE = 42

def find_optimal_clusters(data, max_clusters=10):
    scores = []
    for k in range(2, max_clusters + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(data)
        scores.append(silhouette_score(data, labels))
    return scores.index(max(scores)) + 2

def plot_clusters(ax, points, labels, centroids, title, iteration=None):
    ax.clear()
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Главная компонента 1')
    ax.set_ylabel('Главная компонента 2')
    ax.grid(True, alpha=0.3)

    for i in range(len(centroids)):
        cluster_points = points[labels == i]
        if len(cluster_points):
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       s=25, c=COLORS[i % len(COLORS)], alpha=0.7, label=f'Кластер {i + 1}')

    ax.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='X',
               c='black', edgecolors='yellow', linewidth=2, label='Центроиды')
    ax.legend(loc='upper right', fontsize=8)


def kmeans_with_viz(data, k, max_iters=100, tol=1e-4, pause=1.2):
    np.random.seed(RANDOM_STATE)

    idx = np.random.choice(len(data), k, replace=False)
    centroids = data[idx].copy()

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))

    with tqdm(range(max_iters), desc="K-means optimization") as pbar:
        for step in pbar:
            distances = np.array([[np.linalg.norm(p - c) for c in centroids] for p in data])
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([data[labels == i].mean(axis=0) if len(data[labels == i]) > 0
                                      else centroids[i] for i in range(k)])

            iter_info = f'Итерация {step + 1} из {max_iters}'
            plot_clusters(ax, data, labels, centroids, iter_info)

            if step == 0:
                plt.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='X',
                            c='black', edgecolors='yellow', linewidth=2)
            plt.pause(pause)

            if np.linalg.norm(new_centroids - centroids) <= tol:
                plot_clusters(ax, data, labels, new_centroids, 'Финальная кластеризация')
                ax.scatter(new_centroids[:, 0], new_centroids[:, 1], s=400, marker='X',
                           c='darkred', edgecolors='white', linewidth=3, label='Финальные центры')
                ax.legend(loc='upper right', fontsize=8)
                plt.pause(3)
                pbar.set_description(f"Сошлось на шаге {step + 1}")
                break

            centroids = new_centroids
    plt.ioff()
    return centroids, labels

def run_pipeline():
    print("K-MEANS")
    print("Обработка набора данных Iris")

    data, true_labels = load_iris(return_X_y=True)
    data_scaled = StandardScaler().fit_transform(data)

    optimal_k = find_optimal_clusters(data_scaled)
    print(f"Оптимальное количество кластеров: {optimal_k}")

    data_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(data_scaled)

    final_centroids, labels = kmeans_with_viz(data_2d, optimal_k)

    silhouette_avg = silhouette_score(data_2d, labels)
    print(f"Силуэтный коэффициент: {silhouette_avg:.3f}")
    print(f"Количество точек: {len(labels)}")
    print(f"Распределение по кластерам: {np.bincount(labels)}")

    plt.show()

    print("\n")
    print("Программа завершила работу")
    print(f"Координаты финальных центроидов:\n{final_centroids}")
    print("\n")

    return final_centroids, labels


if __name__ == "__main__":
    cluster_centers, cluster_labels = run_pipeline()