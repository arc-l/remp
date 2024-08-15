import math
from typing import List
import os

import rvg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

import ompl.base as ob
import ompl.geometric as og
import cv2
from constants import TRANS_WEIGHT, ROT_WEIGHT, BOUNDARY

# TODO: change this in source code for speed
class CustomSE2StateSpace(ob.SE2StateSpace):
    def distance(self, state1, state2):
        # compute the Euclidean distance between the x-y components of the states
        xy_distance = super().distance(state1, state2)

        # Calculate the angular distance between yaw angles (in radians)
        angular_distance = abs(state2.getYaw() - state1.getYaw())
        if angular_distance > math.pi:
            angular_distance = 2 * math.pi - angular_distance
        # assert angular_distance >= 0

        # Combine position and angular distances into a single cost
        # You can use different weights for position and angular distances depending on your problem
        # total_distance = TRANS_WEIGHT * xy_distance + ROT_WEIGHT * angular_distance
        total_distance = max(TRANS_WEIGHT * xy_distance, ROT_WEIGHT * angular_distance)

        return total_distance



def shapely_collision_check(object: Polygon, obstacles: List[Polygon]) -> bool:
    """Check if the object collides with any of the obstacles"""
    for obs in obstacles:
        if object.intersects(obs):
            return True
    return False

def shapely_collision_check_ids(obj_id: int, obj_poly: Polygon, polys: List[Polygon]) -> List[int]:
    """Check if the object collides with any of the obstacles"""

    ids = []
    for i, obs in enumerate(polys):
        if i == obj_id:
            continue
        if obj_poly.intersects(obs):
            ids.append(i)
    return ids


def shapely_get_center(geom: BaseGeometry) -> List[float]:
    """Take from shapely.affinity.interpret_origin"""
    minx, miny, maxx, maxy = geom.bounds
    origin = [(maxx + minx) / 2.0, (maxy + miny) / 2.0]

    return origin


def shapely_rotate_with_center(geom: BaseGeometry, angle: float, x0: float, y0: float) -> Polygon:
    """Take from shapely.affinity.rotate"""
    cosp = math.cos(angle)
    sinp = math.sin(angle)
    if abs(cosp) < 2.5e-16:
        cosp = 0.0
    if abs(sinp) < 2.5e-16:
        sinp = 0.0

    # fmt: off
    matrix = (
        cosp, -sinp, 0.0,
        sinp, cosp, 0.0,
        0.0, 0.0, 1.0,
        x0 - x0 * cosp + y0 * sinp, y0 - x0 * sinp - y0 * cosp, 0.0
    )
    # fmt: on
    return affine_transform(geom, matrix)

def shapely_translate(geom: BaseGeometry, xoff: float, yoff: float) -> Polygon:
    """Take from shapely.affinity.translate"""
    if geom.is_empty:
        return geom

    # fmt: off
    matrix = (1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0,
              xoff, yoff, 0.0)
    # fmt: on
    return affine_transform(geom, matrix)

def shapely_rotate_translate_with_center(
    geom: BaseGeometry, xoff: float, yoff: float, angle: float, x0: float, y0: float
) -> Polygon:
    """Take from shapely.affinity.rotate and shapely.affinity.translate"""
    cosp = math.cos(angle)
    sinp = math.sin(angle)
    if abs(cosp) < 2.5e-16:
        cosp = 0.0
    if abs(sinp) < 2.5e-16:
        sinp = 0.0

    # fmt: off
    matrix = (
        cosp, -sinp, 0.0,
        sinp, cosp, 0.0,
        0.0, 0.0, 1.0,
        x0 - x0 * cosp + y0 * sinp + xoff, y0 - x0 * sinp - y0 * cosp + yoff, 0.0
    )
    # fmt: on
    return affine_transform(geom, matrix)

# def compute_eigen_vectors(polygon):
#     points = np.array(polygon.exterior.coords[:-1])
#     mean, eigenvectors = cv2.PCACompute(points, mean=None)
#     return eigenvectors[0]
def compute_long_axis(polygon):
    points = np.array(polygon.exterior.coords[:-1], dtype=np.float32)
    rect = cv2.minAreaRect(points)
    # rect contains: (center (x,y), (width, height), angle of rotation)
    (center_x, center_y), (width, height), angle = rect
    # If the width is less than the height, then the angle is off by 90 degrees
    if width < height:
        angle += 90
    return np.radians(angle)

def compute_long_axis_radius(polygon):
    points = np.array(polygon.exterior.coords[:-1], dtype=np.float32)
    rect = cv2.minAreaRect(points)
    # rect contains: (center (x,y), (width, height), angle of rotation)
    (center_x, center_y), (width, height), angle = rect
    # If the width is less than the height, then the angle is off by 90 degrees
    if width < height:
        angle += 90
    return np.radians(angle), min(width, height) / 2, max(width, height) / 2, np.array([center_x, center_y])

def plot_polygons(polygons, filename="test"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(BOUNDARY[0, 0], BOUNDARY[0, 1])
    ax.set_ylim(BOUNDARY[1, 0], BOUNDARY[1, 1])

    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        text_x, text_y = polygon.centroid.x, polygon.centroid.y
        ax.text(text_x, text_y, str(i))
        ax.plot(x, y, linewidth=1, zorder=2)
        ax.fill(x, y, alpha=0.3, zorder=2)

    ax.set_title("Polygons from Binary Image")
    # plt.show()
    plt.savefig(f"{filename}.png")

def plot_polygons_with_label(polygons, action_types, filename="test"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(BOUNDARY[0, 0], BOUNDARY[0, 1])
    ax.set_ylim(BOUNDARY[1, 0], BOUNDARY[1, 1])
    default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color'] * 20

    for i, (polygon, color) in enumerate(zip(polygons, default_colors)):
        x, y = polygon.exterior.xy
        # text_x, text_y = polygon.centroid.x - 0.01, polygon.centroid.y - 0.01
        # t = "pp" if action_types[i] == 0 else "pt"
        # ax.text(text_x, text_y, str(i))
        
        # if i >= 4:
        #     ax.plot(x, y, linewidth=1, color=color)
        if i < 4:
            # if action_types[i] == 0:
            #     ax.fill(x, y, alpha=0.05)
            # else:
            ax.fill(x, y, alpha=0.7)
        else:
            ax.plot(x, y, linewidth=1, color=default_colors[1])
        


    ax.set_title("Polygons from Binary Image")
    # plt.show()
    plt.savefig(f"{filename}.png")

def plot_polygons_with_direction(polygons, filename="test"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    # Assuming BOUNDARY is defined elsewhere, or you can define it manually.
    ax.set_xlim(BOUNDARY[0, 0], BOUNDARY[0, 1])
    ax.set_ylim(BOUNDARY[1, 0], BOUNDARY[1, 1])
    
    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        text_x, text_y = polygon.centroid.x, polygon.centroid.y
        ax.text(text_x, text_y, str(i))
        ax.plot(x, y, linewidth=1)

        # # Calculate the eigenvectors and eigenvalues of the current polygon
        # centroid = np.array([text_x, text_y])
        # points = np.array(polygon.exterior.coords[:-1])
        # mean, eigenvectors = cv2.PCACompute(points, mean=None)
        
        # # Plot the eigenvector as an arrow originating from the centroid
        # for eigenvector in eigenvectors.T:
        #     scale_factor = 0.05  # Adjust this factor to control the length of the arrow
        #     arrow_end = centroid + scale_factor * eigenvector
        #     ax.arrow(centroid[0], centroid[1], arrow_end[0] - centroid[0], arrow_end[1] - centroid[1],
        #              head_width=0.01, head_length=0.01, fc='red', ec='red')

        # find the minimum area rectangle that encloses the polygon
        points = np.array(polygon.exterior.coords[:-1], dtype=np.float32)
        rect = cv2.minAreaRect(points)
        # rect contains: (center (x,y), (width, height), angle of rotation)
        (center_x, center_y), (width, height), angle = rect
        # If the width is less than the height, then the angle is off by 90 degrees
        if width < height:
            angle += 90

        # box = cv2.boxPoints(rect)
        # ax.plot(box[:,0], box[:,1], 'r')
        ax.plot([center_x - np.cos(np.radians(angle)) * width / 2, center_x + np.cos(np.radians(angle)) * width / 2],
            [center_y - np.sin(np.radians(angle)) * width / 2, center_y + np.sin(np.radians(angle)) * width / 2], 'b')

    ax.set_title("Polygons from Binary Image with Eigenvectors")
    plt.savefig(f"{filename}.png")

def plot_polygons_with_drag(polygons, poly, path, filename="test"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(BOUNDARY[0, 0], BOUNDARY[0, 1])
    ax.set_ylim(BOUNDARY[1, 0], BOUNDARY[1, 1])

    for i, polygon in enumerate(polygons):
        x, y = polygon.exterior.xy
        text_x, text_y = polygon.centroid.x, polygon.centroid.y
        ax.text(text_x, text_y, str(i))
        ax.plot(x, y, linewidth=1)

    for p in path:
        new_poly = shapely_rotate_translate_with_center(
            poly, p[0], p[1], p[2], 0, 0
        )
        x, y = new_poly.exterior.xy

        if p == path[-1]:
            ax.plot(x, y, linewidth=1)
        else:
            ax.fill(x, y, alpha=0.1, edgecolor='gray', linewidth=1)
    

    ax.set_title("Polygons from Binary Image")
    # plt.show()
    plt.savefig(f"{filename}.png")

def extrude_polygon(polygon: List[float], height: float, file_name: str):

    # Create vertices
    vertices = []

    # Lower vertices (z = 0)
    for point in polygon:
        vertices.append([point[0], point[1], -height / 2])

    # Upper vertices (z = height)
    for point in polygon:
        vertices.append([point[0], point[1], height / 2])

    # Create indices
    indices = []
    n = len(polygon)

    # Lower face (assumes polygon is convex or star-shaped)
    # Reverse winding to make face visible from outside
    for i in range(1, n-1):
        indices.append([0, i+1, i])

    # Upper face (assumes polygon is convex or star-shaped)
    for i in range(1, n-1):
        indices.append([0+n, i+n, i+1+n])

    # Side faces
    for i in range(n):
        indices.append([i, (i+1)%n, (i+1)%n+n])
        indices.append([i, (i+1)%n+n, i+n])

    # Save a obj file
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write indices
        for index in indices:
            f.write(f"f {index[0] + 1} {index[1] + 1} {index[2] + 1}\n")


def angle_between_points(p0, p1, p2):
    v1 = p0 - p1
    v2 = p2 - p1
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    angle = np.arctan2(cross_product, dot_product)
    return np.degrees(angle)

def smoothen_sharp_corners(polygon, angle_threshold=45):
    smoothed_polygon = []
    num_vertices = len(polygon)

    for i in range(num_vertices):
        p0 = polygon[i-1][0]
        p1 = polygon[i][0]
        p2 = polygon[(i+1)%num_vertices][0]
        angle = angle_between_points(p0, p1, p2)
        if abs(angle) < angle_threshold:
            midepoint = (p0 + p2) / 2
            smoothed_polygon.append(midepoint)
        else:
            smoothed_polygon.append(p1)

    return np.array(smoothed_polygon, dtype=polygon.dtype).reshape((-1, 1, 2))

def shapely_polygon_to_rvg_polygon(polygon: Polygon):
    x, y = polygon.exterior.xy
    rvg_polygon = [
        rvg.vertex(x[i], y[i])
        for i in range(len(x)-1)
    ]
    rvg_polygon = rvg.polygon(rvg_polygon, False)
    return rvg_polygon