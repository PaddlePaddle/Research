import numpy as np
import cv2
import math

# map [0,255] into 8 section
def bgr_mapping(img_val):
    if img_val >= 0 and img_val <= 31: return 0
    if img_val >= 32 and img_val <= 63: return 1
    if img_val >= 64 and img_val <= 95: return 2
    if img_val >= 96 and img_val <= 127: return 3
    if img_val >= 128 and img_val <= 159: return 4
    if img_val >= 160 and img_val <= 191: return 5
    if img_val >= 192 and img_val <= 223: return 6
    if img_val >= 224: return 7

# Calculate color histogram
def calc_bgr_hist(image):
    if not image.size: return False
    hist = {}
    image = cv2.resize(image, (32, 32)) # resize img decrease computation
    for bgr_list in image:
        for bgr in bgr_list:
            maped_b = bgr_mapping(bgr[0])
            maped_g = bgr_mapping(bgr[1])
            maped_r = bgr_mapping(bgr[2])
            index = maped_b * 8 * 8 + maped_g * 8 + maped_r
            hist[index] = hist.get(index, 0) + 1

    return hist

# Calculate color histogram similarity
def compare_similar_hist(h1, h2):
    if not h1 or not h2: return False
    sum1, sum2, sum_mixd = 0, 0, 0
    for i in range(512):
        sum1 = sum1 + (h1.get(i, 0) * h1.get(i, 0))
        sum2 = sum2 + (h2.get(i, 0) * h2.get(i, 0))
        sum_mixd = sum_mixd + (h1.get(i, 0) * h2.get(i, 0))
    # cosine similarity
    return sum_mixd / (math.sqrt(sum1) * math.sqrt(sum2))

# Calculate color psnr similarity
def psnr(img1, img2):
    img1 = cv2.resize(img1, (10, 10))
    img2 = cv2.resize(img2, (10, 10))
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

d = [[-1, 0], [1, 0], [0, 1], [0, -1]]
# search connected region
def search_region(G, pos):
    x1, y1, x2, y2 = pos[1], pos[0], pos[1], pos[0]
    Q = set()
    Q.add(pos)
    h, w = G.shape
    visited = np.zeros((h, w))
    visited[pos] = 1
    while Q:
        u = Q.pop()
        for move in d:
            row = u[0] + move[0]
            col = u[1] + move[1]
            if (row >= 0 and row < h and col >= 0 and col < w and G[row, col] == 1 and visited[row, col] == 0):
                visited[row, col] = 1
                Q.add((row, col))
                x1 = min(x1, col)
                x2 = max(x2, col)
                y1 = min(y1, row)
                y2 = max(y2, row)
    return [int(x1), int(y1), int(x2), int(y2)]

def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        # print(intersect,sum_area)
        return (float(intersect) / float(sum_area - intersect)) * 1.0
