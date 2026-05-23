import pygame
import math

WIDTH, HEIGHT = 800, 600
EPS = 20
MIN_SAMPLES = 3

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

def get_neighbors(points, target_idx, eps):
    neighbors = []
    p1 = points[target_idx]
    for i, p2 in enumerate(points):
        if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < eps:
            neighbors.append(i)
    return neighbors


def update_dbscan(points):
    n = len(points)
    if n == 0:
        return []

    labels = [None] * n
    visited = [False] * n

    main_cluster_found = False

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = get_neighbors(points, i, EPS)

        if len(neighbors) < MIN_SAMPLES:
            labels[i] = -1
        else:
            if not main_cluster_found:
                main_cluster_found = True
                labels[i] = 0

                queue = neighbors[:]
                for idx in queue:
                    if not visited[idx]:
                        visited[idx] = True
                        new_neighbors = get_neighbors(points, idx, EPS)
                        if len(new_neighbors) >= MIN_SAMPLES:
                            queue.extend([idx_n for idx_n in new_neighbors if idx_n not in queue])
                            labels[idx] = 0
                        else:
                            labels[idx] = 1
                    elif labels[idx] == -1 or labels[idx] is None:
                        labels[idx] = 1
            else:
                labels[i] = -1
    return [l if l is not None else -1 for l in labels]


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DBSCAN: Green=Cluster, Yellow=Boundary, Red=Noise")
    clock = pygame.time.Clock()

    points = []
    labels = []
    running = True

    while running:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    points = []
                    labels = []

        mouse_pressed = pygame.mouse.get_pressed()
        if mouse_pressed[0]:
            pos = pygame.mouse.get_pos()
            if not points or math.hypot(pos[0] - points[-1][0], pos[1] - points[-1][1]) > 5:
                points.append(pos)

        if points:
            labels = update_dbscan(points)

        for i, point in enumerate(points):
            color = RED
            if i < len(labels):
                if labels[i] == 0:
                    color = GREEN
                elif labels[i] == 1:
                    color = YELLOW

            pygame.draw.circle(screen, color, point, 6)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()