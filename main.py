import cv2
import numpy as np
import glob
import os
import math
from PIL import Image

class DisjointSet:
    __slots__ = ('parent', 'rank', 'elements')

    def __init__(self, n, m):
        self.parent = {(i, j): (i, j) for i in range(n) for j in range(m)}
        self.rank = {(i, j): 0 for i in range(n) for j in range(m)}
        self.elements = {(i, j): [] for i in range(n) for j in range(m)}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def add_to_set(self, i, j):
        root = self.find((i, j))
        self.elements[root].append((i, j))

    def get_all_sets(self):
        all_sets = {}
        for idx, root in self.parent.items():
            root = self.find(idx)
            if root not in all_sets:
                all_sets[root] = []
            all_sets[root].append(idx)
        return all_sets

def is_within_distance(x1, y1, x2, y2, max_distance=10.0):
    if (max_distance < 0):
        max_distance = 0
    return (x2 - x1) ** 2 + (y2 - y1) ** 2 <= max_distance ** 2

def preprocess_image(image_path):
    """
    Preprocess the image: convert to grayscale, binarize, and apply morphological operations.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarization using Otsu's thresholding
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite('threshold.png', binary)

    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    return binary

def rectangles_intersect(rect1, rect2):
    """
    Check if two rectangles intersect.
    Each rectangle is defined by [x_min, y_min, x_max, y_max].
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Check for no intersection
    if x1_max < x2_min or x2_max < x1_min:
        return False
    if y1_max < y2_min or y2_max < y1_min:
        return False

    # Otherwise, they intersect
    return True

def merge_rectangles(rect1, rect2):
    """
    Merge two intersecting rectangles into one bounding rectangle.
    """
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Create a bounding rectangle that encompasses both
    return [
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max),
    ]

def group_intersecting_rectangles(rect_dict):
    """
    Group intersecting rectangles, allowing for iterative merging.
    rect_dict: A dictionary where keys are linear indices and values are rectangles.
    """
    rects = list(rect_dict.values())
    merged = True

    while merged:
        merged = False
        new_rects = []
        skip_indices = set()

        for i in range(len(rects)):
            if i in skip_indices:
                continue

            current_rect = rects[i]
            for j in range(i + 1, len(rects)):
                if j in skip_indices:
                    continue

                # Check if current_rect and rects[j] intersect
                if rectangles_intersect(current_rect, rects[j]):
                    # Merge them and skip adding rects[j]
                    current_rect = merge_rectangles(current_rect, rects[j])
                    skip_indices.add(j)
                    merged = True

            new_rects.append(current_rect)

        # Replace rects with the merged result
        rects = new_rects

    # Create the final dictionary with grouped rectangles
    # result = {i + 1: rect for i, rect in enumerate(rects)}
    return rects

def determine_orientation_new(centers, corners, base_tolerance=0.05, scaling_factor=2.5, classification_threshold=1):
    """
    Determines the orientation of text (horizontal or vertical) based on bounding box centers and corners.

    Parameters:
        centers (list of tuples): List of (x, y) coordinates representing the centers of bounding boxes.
        corners (list of tuples): List of ((lowerX, lowerY), (upperX, upperY)) representing bounding box corners.
        base_tolerance (float): Base multiplier for primary grouping tolerance.
        scaling_factor (float): Scaling factor for secondary tolerance.
        classification_threshold (float): Percentage threshold for determining orientation via size heuristic.

    Returns:
        str: "horizontal" or "vertical".
    """
    def calculate_dynamic_tolerances(corners):
        """Calculate dynamic primary and secondary tolerances based on bounding box dimensions."""
        widths = [upperX - lowerX for (lowerX, lowerY), (upperX, upperY) in corners]
        heights = [upperY - lowerY for (lowerX, lowerY), (upperX, upperY) in corners]
        #print("Widths: ", widths)
        #print("Heights: ", heights)

        avg_width = np.mean(widths) if widths else 0
        avg_height = np.mean(heights) if heights else 0
        #print("Avg Width: ", avg_width)
        #print("Avg Height: ", avg_height)

        primary_tolerance_horizontal = base_tolerance * avg_height  # Based on average height for horizontal grouping
        primary_tolerance_vertical = base_tolerance * avg_width    # Based on average width for vertical grouping
        #print("Primary Tolerance Horizontal: ", primary_tolerance_horizontal)
        #print("Primary Tolerance Vertical: ", primary_tolerance_vertical)

        secondary_tolerance_horizontal = scaling_factor * avg_width
        secondary_tolerance_vertical = scaling_factor * avg_height
        #print("Secondary Tolerance Horizontal: ", secondary_tolerance_horizontal)
        #print("Secondary Tolerance Vertical: ", secondary_tolerance_vertical)

        return (primary_tolerance_horizontal, primary_tolerance_vertical,
                secondary_tolerance_horizontal, secondary_tolerance_vertical)

    def bounding_box_size_heuristic(corners):
        """Classify orientation based on bounding box size and identify outliers."""
        horizontal_outliers = 0
        vertical_outliers = 0
        total_boxes = len(corners)

        filtered_indices = []  # Indices of boxes to keep for spacing heuristic

        for i, ((lowerX, lowerY), (upperX, upperY)) in enumerate(corners):
            width = upperX - lowerX
            height = upperY - lowerY

            if width > 1.5 * height:
                horizontal_outliers += 1
            elif height > 1.5 * width:
                vertical_outliers += 1
            else:
                filtered_indices.append(i)

        if total_boxes == 0:
            return "horizontal", []
        
        horizontal_ratio = horizontal_outliers / total_boxes
        vertical_ratio = vertical_outliers / total_boxes
        if horizontal_ratio > vertical_ratio:
            if horizontal_ratio > classification_threshold:
                return "horizontal", filtered_indices
        elif vertical_ratio > classification_threshold:
            return "vertical", filtered_indices

        return None, filtered_indices

    def group_by_position(centers, primary_index, secondary_index, primary_tolerance, secondary_tolerance):
        """Groups centers based on similarity in primary and secondary positions."""
        clusters = []
        for center in centers:
            placed = False
            for cluster in clusters:
                if (abs(cluster[0][primary_index] - center[primary_index]) <= primary_tolerance and
                    any(abs(other[secondary_index] - center[secondary_index]) <= secondary_tolerance for other in cluster)):
                    cluster.append(center)
                    placed = True
                    break
            if not placed:
                clusters.append([center])
        return clusters

    def calculate_exact_spacing(group, is_horizontal=True):
        """Calculate exact spacing between bounding boxes in a group."""
        if len(group) < 2:
            return None

        # Sort by x coordinate if horizontal, y coordinate if vertical
        index = 0 if is_horizontal else 1
        sorted_centers = sorted(group, key=lambda center: center[index])

        spacings = []
        for i in range(1, len(sorted_centers)):
            prev_center = sorted_centers[i - 1]
            curr_center = sorted_centers[i]

            # Calculate spacing between centers
            if is_horizontal:
                spacing = abs(curr_center[0] - prev_center[0])  # x-coordinate spacing
            else:
                spacing = abs(curr_center[1] - prev_center[1])  # y-coordinate spacing

            spacings.append(spacing)

        return np.mean(spacings) if spacings else None

    # Step 1: Apply size heuristic first
    size_heuristic_result, filtered_indices = bounding_box_size_heuristic(corners)
    if size_heuristic_result:
        return size_heuristic_result

    # Filter centers and corners based on size heuristic
    filtered_centers = [centers[i] for i in filtered_indices]
    filtered_corners = [corners[i] for i in filtered_indices]
    #print("Filtered Centers: ", filtered_centers)
    #print("Filtered Corners: ", filtered_corners)

    # Step 2: Calculate dynamic tolerances
    (primary_tolerance_horizontal, primary_tolerance_vertical,
     secondary_tolerance_horizontal, secondary_tolerance_vertical) = calculate_dynamic_tolerances(filtered_corners)

    # Step 3: Group centers using updated tolerances
    horizontal_clusters = group_by_position(filtered_centers, 1, 0, primary_tolerance_horizontal, secondary_tolerance_horizontal)
    vertical_clusters = group_by_position(filtered_centers, 0, 1, primary_tolerance_vertical, secondary_tolerance_vertical)
    #print("Horizontal Clusters: ", horizontal_clusters)
    #print("Vertical Clusters: ", vertical_clusters)

    # Step 4: Calculate spacing heuristic
    horizontal_spacings = [
        calculate_exact_spacing(cluster, is_horizontal=True) for cluster in horizontal_clusters if len(cluster) > 1
    ]
    vertical_spacings = [
        calculate_exact_spacing(cluster, is_horizontal=False) for cluster in vertical_clusters if len(cluster) > 1
    ]

    avg_horizontal_spacing = np.mean(horizontal_spacings) if horizontal_spacings else 999999
    avg_vertical_spacing = np.mean(vertical_spacings) if vertical_spacings else 999999

    # Step 5: Determine orientation based on spacing heuristic
    return "horizontal" if avg_horizontal_spacing < avg_vertical_spacing else "vertical"

def main(image_path, name, upper, lower):
    # Step 1: Preprocess the image
    binary_image = preprocess_image(f"{image_path}/{name}")
    cv2.imwrite('bin.png', binary_image)

    image = Image.open('bin.png')
    px = image.load()
    n, m = image.width, image.height

    for i in range(n):
        for j in range(m):
            if (px[i, j] > 200):
                px[i, j] = 255
            else:
                px[i, j] = 0
    for i in range(n):
        for j in range(m):
            if (px[i, j] != 255 and px[i, j] != 0):
                print(i, j)

    image.save('filter.png')
    dsu = DisjointSet(image.width, image.height)

    directions = [(i, j) for i in range(-10, 11) for j in range(-10, 11)
                  if is_within_distance(0, 0, i, j, int(math.log(image.width * image.height, 10)-3))]

    for i in range(n):
        for j in range(m):
            if (px[i, j] > 0):
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < m and px[ni, nj] > 0:
                        dsu.union((i, j), (ni, nj))
    
    for i in range(n):
        for j in range(m):
            if px[i, j] > 0:
                dsu.add_to_set(i, j)

    all_chars = dsu.get_all_sets()
    
    bbox = {}
    number = {}
    centers = []
    corners = []

    for root, elements in all_chars.items():
        if (len(elements) > 1):
            min_x = n + 1
            min_y = m + 1
            max_x = -1
            max_y = -1
            for item in elements:
                x, y = item
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            bbox[root] = [min_x, min_y, max_x, max_y]
            number[root] = len(elements)
    
    with open("bbox.txt", "w") as f:
        f.write(f"{n} {m}\n")
        for root in bbox:
            f.write(f"{root}: {bbox[root]}\n")

    test = Image.new('RGB', (n, m))
    test_new = Image.new('RGB', (n, m))
    for i in range(n):
        for j in range(m):
            test.putpixel((i, j), (px[i, j], px[i, j], px[i, j], 255))
            test_new.putpixel((i, j), (px[i, j], px[i, j], px[i, j], 255))

    filter_bbox = {}

    for root in bbox:
        min_x, min_y, max_x, max_y = bbox[root]
        area = (max_x + 1 - min_x) * (max_y + 1 - min_y)

        if (area > upper * m * n or area < lower * m * n):
            continue

        for i in range(min_x, max_x + 1):
            test.putpixel((i, min_y), (255, 0, 0, 255))
            test.putpixel((i, max_y), (255, 0, 0, 255))    
        for i in range(min_y, max_y + 1):
            test.putpixel((min_x, i), (255, 0, 0, 255))
            test.putpixel((max_x, i), (255, 0, 0, 255))
        filter_bbox[root] = bbox[root]

    new_bbox = group_intersecting_rectangles(filter_bbox)
    with open("new_bbox.txt", "w") as f:
        f.write(f"{n} {m}\n")
        for root in new_bbox:
            f.write(f"{root}\n")
    for box in new_bbox:
        min_x, min_y, max_x, max_y = box
        area = (max_x + 1 - min_x) * (max_y + 1 - min_y)
        if (area > upper * m * n or area < lower * m * n):
            continue
        
        corners.append(((min_x, min_y),(max_x, max_y)))

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        centers.append((center_x, center_y))
        
        for i in range(min_x, max_x + 1):
            test_new.putpixel((i, min_y), (255, 0, 0, 255))
            test_new.putpixel((i, max_y), (255, 0, 0, 255))    
        for i in range(min_y, max_y + 1):
            test_new.putpixel((min_x, i), (255, 0, 0, 255))
            test_new.putpixel((max_x, i), (255, 0, 0, 255))

    test.save('test.png')
    test_new.save('test_new.png')

    orientation = determine_orientation_new(centers, corners)
    
    print(f"Detected Text Orientation of the image {name}: {orientation}")

    return orientation

def process_images_in_folder(folder_path, upper, lower):
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    cnt = 0
    correct = 0

    # Iterate through the main subfolders (e.g., vertical and horizontal)
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)

        # Ensure it's a directory
        if not os.path.isdir(category_path):
            continue

        print(f"Processing category: {category}")

        # Iterate through sub-subfolders inside the current category
        for subfolder in os.listdir(category_path):
            subfolder_path = os.path.join(category_path, subfolder)

            # Ensure it's a directory
            if not os.path.isdir(subfolder_path):
                continue

            print(f"  Processing subfolder: {subfolder}")

            # Get all image files in the sub-subfolder
            image_files = glob.glob(os.path.join(subfolder_path, "*.*"))

            for image_file in image_files:
                if os.path.splitext(image_file)[1].lower() in supported_extensions:
                    # Full file path and name of the image
                    name = os.path.basename(image_file)
                    print(f"    Processing file: {image_file}...")  # Show full path of the image

                    # Call the main function
                    cnt += 1
                    result = main(subfolder_path, name, upper, lower)

                    # Check if the result matches the category (e.g., "vertical" or "horizontal")
                    if result == category.lower():  # Match with the folder name
                        correct += 1

                    # Print accuracy after processing each image
                    accuracy = correct / cnt
                    print(f"    Accuracy after {cnt} iterations: {accuracy:.2%} ({correct}/{cnt})")

data_path = "./data/"   # Path for data

upper = 0.2       # Percentage area to the image size to upper skip bounding
lower = 0.000008       # Percentage area to the image size to lower skip bounding

process_images_in_folder(data_path, upper, lower)