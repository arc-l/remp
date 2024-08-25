import multiprocessing
import os
import rvg
import matplotlib.pyplot as plt
import random
from typing import List
import time
import math
from functools import partial
from tqdm import tqdm

import numpy as np
from shapely.geometry import Polygon, box, Point
from shapely.affinity import translate
from shapely.ops import unary_union
import ompl.base as ob
import ompl.geometric as og
import ompl.util

ompl.util.setLogLevel(ompl.util.LogLevel.LOG_WARN)  # type: ignore

from mcts.node import Node, Action
from constants import *
from utils import shapely_collision_check, CustomSE2StateSpace, shapely_rotate_translate_with_center, compute_long_axis_radius, shapely_polygon_to_rvg_polygon

import signal
def handler(signum, frame):
    raise TimeoutError("RRT solve didn't finish in time")
signal.signal(signal.SIGALRM, handler)

manager = multiprocessing.Manager()
counter = manager.Value("i", 0)
# pids = manager.list()
lock = manager.Lock()


def is_obj_at_goal(obj_pose: np.ndarray, goal_pose: np.ndarray) -> bool:
    diff = obj_pose[2] - goal_pose[2]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return (
        math.isclose(obj_pose[0], goal_pose[0], abs_tol=IS_CLOSE_TRANS_THRESHOLD)
        and math.isclose(obj_pose[1], goal_pose[1], abs_tol=IS_CLOSE_TRANS_THRESHOLD)
        and math.isclose(diff, 0, abs_tol=IS_CLOSE_ROT_THRESHOLD)
    )


def get_pick_place_cost(obj_pose: np.ndarray, goal_pose: np.ndarray) -> float:
    """Calculate the cost of pick-and-place action"""
    trans_dist = math.sqrt((obj_pose[0] - goal_pose[0]) ** 2 + (obj_pose[1] - goal_pose[1]) ** 2)
    rot_dist = abs(obj_pose[2] - goal_pose[2])
    if rot_dist > math.pi:
        rot_dist = 2 * math.pi - rot_dist
    assert rot_dist >= 0
    # cost = PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (trans_dist * TRANS_WEIGHT + rot_dist * ROT_WEIGHT)
    cost = max(PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (trans_dist * TRANS_WEIGHT), PICK_PLACE_BASE_COST + PICK_PLACE_DIST_SCALE * (rot_dist * ROT_WEIGHT))

    return cost


def sample_next_poses(
    curr_pose: np.ndarray,
    goal_pose: np.ndarray,
    grid_action: np.ndarray,
    inner_radius: float,
    outer_radius: float,
    center_offset: float,
    long_angle: float,
    is_simulate: bool = False,
    grid_sample_half: bool = False,
) -> np.ndarray:
    """Sample next poses for the object, all poses are within the boundary"""
    at_goal = is_obj_at_goal(curr_pose, goal_pose)

    # sample poses around the goal pose
    if is_simulate or at_goal:
        num_trans_samples = 7
        num_rot_samples = 1
    else:
        num_trans_samples = 10
        num_rot_samples = 1
    trans_low, trans_high = 0.03, 0.15
    # trans_low, trans_high = inner_radius, inner_radius + 0.1
    signs = np.random.choice([-1, 1], size=(num_trans_samples, 2))
    trans_offset = np.random.uniform(low=trans_low, high=trans_high, size=(num_trans_samples, 2)) * signs
    new_pos = trans_offset + goal_pose[0:2]
    new_pos[:, 0] = np.clip(new_pos[:, 0], BOUNDARY[0, 0] + inner_radius, BOUNDARY[0, 1] - inner_radius)
    new_pos[:, 1] = np.clip(new_pos[:, 1], BOUNDARY[1, 0] + inner_radius, BOUNDARY[1, 1] - inner_radius)
    new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
    new_pose_goal = np.hstack((new_pos, angles))

    # sample poses within the boundary
    if is_simulate:
        num_trans_samples = 7
        num_rot_samples = 1
    else:
        num_trans_samples = 16
        num_rot_samples = 1
    new_pos = np.random.uniform(
        low=BOUNDARY[:2, 0] + inner_radius, high=BOUNDARY[:2, 1] - inner_radius, size=(num_trans_samples, 2)
    )
    new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
    angles = np.random.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
    new_pose_rand = np.hstack((new_pos, angles))

    # sample poses as a grid over the boundary
    if is_simulate:
        new_pose_grid = sample_grid_actions(inner_radius, outer_radius, center_offset, long_angle)
        if grid_sample_half:
            new_pose_grid = new_pose_grid[np.random.choice(len(new_pose_grid), len(new_pose_grid) // 4, replace=False)]
        else:
            new_pose_grid = new_pose_grid[np.random.choice(len(new_pose_grid), len(new_pose_grid) // 2, replace=False)]
    else:
        new_pose_grid = grid_action
        if grid_sample_half:
            new_pose_grid = new_pose_grid[np.random.choice(len(new_pose_grid), len(new_pose_grid) // 2, replace=False)]

    # sample poses around the current pose
    if not at_goal:
        if is_simulate:
            num_trans_samples = 5
            num_rot_samples = 1
        else:
            num_trans_samples = 7
            num_rot_samples = 1
        trans_low, trans_high = 0.03, 0.15
        # trans_low, trans_high = inner_radius, inner_radius + 0.1
        signs = np.random.choice([-1, 1], size=(num_trans_samples, 2))
        trans_offset = np.random.uniform(low=trans_low, high=trans_high, size=(num_trans_samples, 2)) * signs
        new_pos = trans_offset + curr_pose[0:2]
        new_pos[:, 0] = np.clip(new_pos[:, 0], BOUNDARY[0, 0] + inner_radius, BOUNDARY[0, 1] - inner_radius)
        new_pos[:, 1] = np.clip(new_pos[:, 1], BOUNDARY[1, 0] + inner_radius, BOUNDARY[1, 1] - inner_radius)
        new_pos = np.repeat(new_pos, num_rot_samples, axis=0)
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(len(new_pos), 1))
        new_pose_curr = np.hstack((new_pos, angles))
        next_poses = np.vstack((new_pose_goal, new_pose_curr, new_pose_rand, new_pose_grid))
    else:
        next_poses = np.vstack((new_pose_goal, new_pose_rand, new_pose_grid))

    return next_poses


def sample_single_obj_actions(
    curr_pose: np.ndarray,
    goal_pose: np.ndarray,
    grid_action: np.ndarray,
    obj_id: int,
    action_type: int,
    obj_inner_radius: float,
    obj_outer_radius: float,
    center_offset: float,
    long_angle: float,
    grid_sample_half: bool = False,
) -> List[Action]:
    """Get all possible actions for the object"""
    actions = []
    next_poses = sample_next_poses(curr_pose, goal_pose, grid_action, obj_inner_radius, obj_outer_radius, center_offset, long_angle, grid_sample_half)
    for next_pose in next_poses:
        actions.append(Action(action_type, obj_id, next_pose))

    return actions


def sample_actions(
    obj_num: int,
    obj_action_types: List[int],
    obj_poses: np.ndarray,
    goal_poses: np.ndarray,
    grid_actions: List[np.ndarray],
    obj_inner_radii: List[float],
    obj_outer_radii: List[float],
    obj_center_offsets: List[float],
    obj_long_angles: List[float],
    grid_sample_half: bool = False,
) -> List[Action]:
    """Get all possible actions for the node, which is a list of actions for each object"""
    actions = []
    goal_actions = []

    for obj_id in range(obj_num):
        action_type = obj_action_types[obj_id]
        curr_pose = obj_poses[obj_id]
        goal_pose = goal_poses[obj_id]

        if not is_obj_at_goal(curr_pose, goal_pose):
            goal_actions.append(Action(action_type, obj_id, goal_pose))

        sub_actions = sample_single_obj_actions(
            curr_pose, goal_pose, grid_actions[obj_id], obj_id, action_type, obj_inner_radii[obj_id], obj_outer_radii[obj_id], obj_center_offsets[obj_id], obj_long_angles[obj_id], grid_sample_half
        )
        if len(sub_actions) == 0:
            continue
        actions.extend(sub_actions)

    random.shuffle(actions)
    random.shuffle(goal_actions)

    # prioritize goal actions
    actions = goal_actions + actions

    return actions


def sample_grid_actions(short_radius: float, long_radius: float, center_offset: float, long_angle: float) -> np.ndarray:
    """Sample poses as a grid over the boundary"""
    rot_half = (short_radius / long_radius > 0.9)
    max_nx, max_ny = 8, 5
    max_radius = 0.25

    # make grid horizontal
    nx = round((BOUNDARY[0, 1] - BOUNDARY[0, 0]) / (max_radius - long_radius))
    ny = round((BOUNDARY[1, 1] - BOUNDARY[1, 0]) / (max_radius - short_radius))
    nx = min(max_nx, nx)
    ny = min(max_ny, ny)
    x_min, x_max = BOUNDARY[0, 0] + long_radius, BOUNDARY[0, 1] - long_radius
    y_min, y_max = BOUNDARY[1, 0] + short_radius, BOUNDARY[1, 1] - short_radius
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)
    new_pos = np.column_stack((xx.ravel(), yy.ravel()))
    # center it
    # new_pos = new_pos - center_offset
    new_pos[:, 0] = new_pos[:, 0] - center_offset[0]
    new_pos[:, 1] = new_pos[:, 1] - center_offset[1]
    angles = np.array([-long_angle])
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    angles = np.tile(angles, len(new_pos))
    new_pose_grid_hori = np.c_[new_pos, angles]
    new_pose_grid = new_pose_grid_hori

    # make grid vertical
    if not rot_half:
        nx = round((BOUNDARY[0, 1] - BOUNDARY[0, 0]) / (max_radius - short_radius))
        ny = round((BOUNDARY[1, 1] - BOUNDARY[1, 0]) / (max_radius - long_radius))
        nx = min(max_nx, nx)
        ny = min(max_ny, ny)
        x_min, x_max = BOUNDARY[0, 0] + short_radius, BOUNDARY[0, 1] - short_radius
        y_min, y_max = BOUNDARY[1, 0] + long_radius, BOUNDARY[1, 1] - long_radius
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        xx, yy = np.meshgrid(x, y)
        new_pos = np.column_stack((xx.ravel(), yy.ravel()))
        # center it
        new_pos[:, 0] = new_pos[:, 0] + center_offset[1]
        new_pos[:, 1] = new_pos[:, 1] - center_offset[0]
        angles = np.array([-long_angle + np.pi / 2])
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        angles = np.tile(angles, len(new_pos))
        new_pose_grid_vert = np.c_[new_pos, angles]
        new_pose_grid = np.vstack((new_pose_grid, new_pose_grid_vert))

    return new_pose_grid


def try_actions_pool(
    actions: List[Action],
    obj_polys: List[Polygon],
    obj_poses: np.ndarray,
    boundary_box,
    goal_poses: np.ndarray,
    cost_so_far: float,
):

    child_nodes = []
    for action in actions:
        obj_poly = obj_polys[action.obj_id]
        obj_pose = obj_poses[action.obj_id]
        if is_obj_at_goal(obj_pose, action.goal_pose):
            continue
        is_valid_action = False
        obs_polys = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
        move = action.goal_pose - obj_pose
        new_obj_poly = shapely_rotate_translate_with_center(
            obj_poly, move[0], move[1], move[2], obj_pose[0], obj_pose[1]
        )
        if new_obj_poly.within(boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
            if action.type == 0:
                cost = get_pick_place_cost(obj_pose, action.goal_pose)
                is_valid_action = True
            else:
                solved, cost, _ = solve_drag_rrt_pool(
                    obj_polys, obj_poses, Action(action.type, action.obj_id, action.goal_pose), boundary_box, optimal=True
                )
                if solved:
                    is_valid_action = True
            if is_valid_action:
                child_obj_polys = obj_polys.copy()
                child_obj_poses = obj_poses.copy()
                child_obj_polys[action.obj_id] = new_obj_poly
                child_obj_poses[action.obj_id] = action.goal_pose
                child_node = Node(
                    child_obj_polys,
                    child_obj_poses,
                    is_goal_state_pool(obj_poses, goal_poses),
                    None,  # pool objects cannot be passed between processes or pickled
                    action,
                    cost_so_far=cost_so_far + cost,
                )
                child_nodes.append(child_node)

    with lock:
        counter.value -= 1
    return child_nodes

def try_actions_pool_map(args):
    actions, obj_polys, obj_poses, boundary_box, goal_poses, cost_so_far = args

    child_nodes = []
    for action in actions:
        obj_poly = obj_polys[action.obj_id]
        obj_pose = obj_poses[action.obj_id]
        if is_obj_at_goal(obj_pose, action.goal_pose):
            continue
        is_valid_action = False
        obs_polys = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
        move = action.goal_pose - obj_pose
        new_obj_poly = shapely_rotate_translate_with_center(
            obj_poly, move[0], move[1], move[2], obj_pose[0], obj_pose[1]
        )
        if new_obj_poly.within(boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
            if action.type == 0:
                cost = get_pick_place_cost(obj_pose, action.goal_pose)
                is_valid_action = True
            else:
                solved, cost, _ = solve_drag_rrt_pool(
                    obj_polys, obj_poses, Action(action.type, action.obj_id, action.goal_pose), boundary_box, optimal=True
                )
                if solved:
                    is_valid_action = True
            if is_valid_action:
                child_obj_polys = obj_polys.copy()
                child_obj_poses = obj_poses.copy()
                child_obj_polys[action.obj_id] = new_obj_poly
                child_obj_poses[action.obj_id] = action.goal_pose
                child_node = Node(
                    child_obj_polys,
                    child_obj_poses,
                    is_goal_state_pool(obj_poses, goal_poses),
                    None,  # pool objects cannot be passed between processes or pickled
                    action,
                    cost_so_far=cost_so_far + cost,
                )
                child_nodes.append(child_node)

    return child_nodes


def is_goal_state_pool(poses: np.ndarray, goal_poses: np.ndarray) -> bool:
    diff = poses[:, 2] - goal_poses[:, 2]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.allclose(poses[:, 0:2], goal_poses[:, 0:2], atol=IS_CLOSE_TRANS_THRESHOLD) and np.all(np.abs(diff) < IS_CLOSE_ROT_THRESHOLD)


def are_objs_at_goal(poses: np.ndarray, goal_poses: np.ndarray) -> np.ndarray:
    """Compute the reward for the state"""
    trans_at_goal = np.isclose(poses[:, 0:2], goal_poses[:, 0:2], atol=IS_CLOSE_TRANS_THRESHOLD)
    trans_at_goal = np.all(trans_at_goal, axis=1)

    diff = poses[:, 2] - goal_poses[:, 2]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    rot_at_goal = np.abs(diff) < IS_CLOSE_ROT_THRESHOLD
    objs_at_goal = np.logical_and(trans_at_goal, rot_at_goal)
    return objs_at_goal.flatten().astype(float)


def solve_drag_rrt_pool(obj_polys: List[Polygon], obj_poses: np.ndarray, action: Action, boundary_box, optimal=False, motion_planner='rrt') -> tuple:
    """Check if the goal pose is reachable"""

    obj_poly = obj_polys[action.obj_id]
    obj_pose = obj_poses[action.obj_id]
    obstacles = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
    rrt_obs_polys = unary_union(obstacles)
    goal_pose = action.goal_pose
    # rotate and translate the object to the origin
    rrt_obj_poly = shapely_rotate_translate_with_center(
        obj_poly, -obj_pose[0], -obj_pose[1], -obj_pose[2], obj_pose[0], obj_pose[1]
    )

    if motion_planner == 'rrt':

        def is_state_valid_rrt_pool(state) -> bool:
            pose = (state.getX(), state.getY(), state.getYaw())
            robot_shape = shapely_rotate_translate_with_center(rrt_obj_poly, pose[0], pose[1], pose[2], 0, 0)
            if not robot_shape.within(boundary_box):
                return False
            if robot_shape.intersects(rrt_obs_polys):
                return False
            return True

        space = CustomSE2StateSpace()
        bounds = ob.RealVectorBounds(2)  # type: ignore
        bounds.setLow(0, BOUNDARY[0, 0])
        bounds.setHigh(0, BOUNDARY[0, 1])
        bounds.setLow(1, BOUNDARY[1, 0])
        bounds.setHigh(1, BOUNDARY[1, 1])
        bounds.setLow(2, -math.pi)
        bounds.setHigh(2, math.pi)
        space.setBounds(bounds)

        rrt_ss = og.SimpleSetup(space)
        rrt_ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_rrt_pool))
        space.setup()
        rrt_ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

        # RRTConnect, LazyLBTRRT, InformedRRTstar, SORRTstar
        planner_fast = og.RRTConnect(rrt_ss.getSpaceInformation())
        planner_fast.setRange(0.3)
        planner_fast.setup()

        planner_optimal = og.RRTConnect(rrt_ss.getSpaceInformation())
        planner_optimal.setRange(0.3)
        planner_optimal.setup()

        start = ob.State(rrt_ss.getStateSpace())
        start().setX(obj_pose[0])
        start().setY(obj_pose[1])
        start().setYaw(obj_pose[2])
        goal = ob.State(rrt_ss.getStateSpace())
        goal().setX(goal_pose[0])
        goal().setY(goal_pose[1])
        goal().setYaw(goal_pose[2])
        rrt_ss.setStartAndGoalStates(start, goal)
        assert is_state_valid_rrt_pool(start()), obj_poses
        assert is_state_valid_rrt_pool(goal()), (obj_poses, goal_pose)
        if not optimal:
            rrt_ss.setPlanner(planner_fast)
        else:
            rrt_ss.setPlanner(planner_optimal)

        if not optimal:
            rrt_ss.solve(0.1)

            if rrt_ss.haveExactSolutionPath():
                path = rrt_ss.getSolutionPath()
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                return True, cost, path
            else:
                return False, None, None
        else:
            rrt_ss.solve(0.3)
            
            if rrt_ss.haveExactSolutionPath():
                path = rrt_ss.getSolutionPath()
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                return True, cost, path
            else:
                return False, None, None
    else:
        obj = shapely_polygon_to_rvg_polygon(rrt_obj_poly)
        boundary = rvg.polygon([
            rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 0]),
            rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 0]),
            rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 1]),
            rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 1]),
            ], False)
        obstacles = [
            shapely_polygon_to_rvg_polygon(obs)
            for obs in obstacles
        ]
        start = rvg.vertex(obj_pose[0], obj_pose[1], 0, 2 * np.pi, obj_pose[2], 2 * np.pi, True)
        goal = rvg.vertex(goal_pose[0], goal_pose[1], 0, 2 * np.pi, goal_pose[2], 2 * np.pi, True)

        # fig, ax = plt.subplots()
        # ax.plot(boundary.getX(), boundary.getY(), '-o', color="darkcyan")
        # for obs in obstacles:
        #     ax.plot(obs.getX(), obs.getY(), '-o', color="darkcyan")
        
        # robot = obj.moveToCopy(start.getX(), start.getY(), start.getTheta())
        # ax.plot(robot.getX(), robot.getY(), '-o', color="deeppink")
        # robot = obj.moveToCopy(goal.getX(), goal.getY(), goal.getTheta())
        # ax.plot(robot.getX(), robot.getY(), '-o', color="darkviolet")
        # plt.show()
        # plt.close(fig)

        try:
            vg = rvg.visibility_graph(robot=obj, border = boundary, obstacles = obstacles, resolution=36, considerSymmetry=True, hashWithTheta=True, simplifiedGeometry=False, numThreads=1, incremental=False, verbose=False)
            vg.setWeight(0.5, 0.5)
        except RuntimeError as e:
            print("Visibility graph failed")

        try:
            path = vg.shortestPath(start, goal)
        except RuntimeError as e:
            print("Shortest path failed")

        if len(path) == 0:
            # print("No path found, return")
            return False, None, None
        else:
            try:
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * vg.getPathLength() 
                # print("Cost found, return")
            except RuntimeError as e:
                print("Get path length failed")
            return True, cost, path


def simulate_pool(
    obj_poses: np.ndarray,
    cost_so_far: float,
    obj_polys: List[Polygon],
    obj_inner_radius: List[float],
    obj_outer_radius: List[float],
    obj_center_offsets: List[float],
    obj_long_angles: List[float],
    depth: int,
    goal_poses: np.ndarray,
    grid_actions: List[np.ndarray],
    goal_reward: float,
    one_obj_goal_reward: float,
    one_obj_push_goal_reward: float,
    simulate_goal_reward_scale: float,
    obj_num: int,
    obj_action_types: List[int],
    boundary_box,
    grid_sample_half: bool,
    max_depth: int,
    max_rollout_steps: int,
    motion_planner: str='rrt',
) -> float:
    """Simulate the environment and return the reward"""

    if is_goal_state_pool(obj_poses, goal_poses):
        with lock:
            counter.value -= 1
        return goal_reward - cost_so_far

    #TODO: deal with this too
    # with lock:
    #     pids.append(os.getpid())

    #TODO: Remove all the checking printouts
    # print(f"check 0 submitted, Pid = {os.getpid()}")
    obj_polys = obj_polys.copy()
    obj_poses = obj_poses.copy()
    cost_so_far = cost_so_far
    start_depth = depth
    obj_id_list = list(range(obj_num))
    obj_at_goal_list = are_objs_at_goal(obj_poses, goal_poses)
    for i in range(len(obj_at_goal_list)):
        if obj_action_types[i] == 1:
            obj_at_goal_list[i] *= one_obj_push_goal_reward
        else:
            obj_at_goal_list[i] *= one_obj_goal_reward
    reward = np.sum(obj_at_goal_list) - cost_so_far
    max_reward = reward
    # print(f"check 1, Pid = {os.getpid()}")

    for _ in range(max_rollout_steps):
        if len(obj_id_list) == 0:
            # print(f"check 4 break, Pid = {os.getpid()}")
            break

        is_valid_action = False

        # print(f"check 7 get in for loop, Pid = {os.getpid()}")
        # priotize goal
        # if random.random() < 0.7:
        if random.random() < max(-0.106 + 0.231 * depth - 0.013 * depth **2, 0.2): 
            # print(f"check 7 priotize goal, Pid = {os.getpid()}")
            test_obj_id_list = obj_id_list.copy()
            random.shuffle(test_obj_id_list)
            for obj_id in test_obj_id_list:
                # print(f"check 9 obj_id = {obj_id}, Pid = {os.getpid()}")
                if obj_at_goal_list[obj_id] > 0:
                    continue
                else:
                    curr_pose = obj_poses[obj_id]
                    action_goal_pose = goal_poses[obj_id]
                    action_type = obj_action_types[obj_id]
                    obj_poly = obj_polys[obj_id]
                    obs_polys = obj_polys[:obj_id] + obj_polys[obj_id + 1 :]
                    
                    move = action_goal_pose - curr_pose
                    new_obj_poly = shapely_rotate_translate_with_center(
                        obj_poly, move[0], move[1], move[2], curr_pose[0], curr_pose[1]
                    )
                    if new_obj_poly.within(boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
                        if action_type == 0:
                            cost = get_pick_place_cost(curr_pose, action_goal_pose)
                            is_valid_action = True
                        else:
                            solved, cost, path = solve_drag_rrt_pool(
                                obj_polys, obj_poses, Action(action_type, obj_id, action_goal_pose), boundary_box, motion_planner=motion_planner
                            )
                            if solved:
                                is_valid_action = True

                        if is_valid_action:
                            obj_polys[obj_id] = new_obj_poly
                            obj_poses[obj_id] = action_goal_pose
                            cost_so_far += cost
                            depth += 1
                            if is_goal_state_pool(obj_poses, goal_poses):
                                new_reward = goal_reward - cost_so_far
                                if new_reward > max_reward:
                                    max_reward = new_reward
                                reward = max(max_reward * 0.5, new_reward)
                                reward = reward * (0.9 ** (depth))
                                with lock:
                                    counter.value -= 1
                                return reward * simulate_goal_reward_scale
                            if action_type == 1:
                                obj_at_goal_list[obj_id] = one_obj_push_goal_reward
                            else:
                                obj_at_goal_list[obj_id] = one_obj_goal_reward
                            new_reward = np.sum(obj_at_goal_list) - cost_so_far
                            if new_reward > max_reward:
                                max_reward = new_reward
                            reward = max(max_reward * 0.5, new_reward)
                            reward = reward * (0.9 ** (depth))
                            obj_id_list = list(range(obj_num))
                            obj_id_list.remove(obj_id)
                            break
        
        # sample an object to move
        if not is_valid_action:    
            # print(f"check 8 sample an object to move, Pid = {os.getpid()}")
            obj_id = random.choice(obj_id_list)
            curr_pose = obj_poses[obj_id]
            goal_pose = goal_poses[obj_id]
            action_type = obj_action_types[obj_id]
            obj_poly = obj_polys[obj_id]
            obs_polys = obj_polys[:obj_id] + obj_polys[obj_id + 1 :]

            # select an action
            action_goal_poses = list(
                sample_next_poses(curr_pose, goal_pose, grid_actions[obj_id], obj_inner_radius[obj_id], obj_outer_radius[obj_id], obj_center_offsets[obj_id], obj_long_angles[obj_id], is_simulate=True, grid_sample_half=grid_sample_half)
            )
            random.shuffle(action_goal_poses)
            # print(f"check 8 shuffled action_goal_poses, Pid = {os.getpid()}")
            cnt = 0
            for action_goal_pose in action_goal_poses:
                cnt+=1
                # print(f"check 8 cnt={cnt}/len{len(action_goal_poses)}, Pid = {os.getpid()}")
                if is_obj_at_goal(curr_pose, action_goal_pose):
                    continue
                # check if the action is valid
                move = action_goal_pose - curr_pose
                new_obj_poly = shapely_rotate_translate_with_center(
                    obj_poly, move[0], move[1], move[2], curr_pose[0], curr_pose[1]
                )
                if new_obj_poly.within(boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
                    if action_type == 0:
                        cost = get_pick_place_cost(curr_pose, action_goal_pose)
                        is_valid_action = True
                    else:
                        # print(f"check 8 solve_drag_rrt_pool, cnt = {cnt}/len{len(action_goal_poses)}, solving, Pid = {os.getpid()}")
                        solved, cost, path = solve_drag_rrt_pool(
                            obj_polys, obj_poses, Action(action_type, obj_id, action_goal_pose), boundary_box, motion_planner=motion_planner
                        )
                        # print(f"check 8 solve_drag_rrt_pool, cnt = {cnt}/len{len(action_goal_poses)}, solved, Pid = {os.getpid()}")
                        if solved:
                            is_valid_action = True

                    # print(f"check 8 is_valid_action={is_valid_action}, Pid = {os.getpid()}")
                    # execute the action
                    if is_valid_action:
                        obj_polys[obj_id] = new_obj_poly
                        obj_poses[obj_id] = action_goal_pose
                        cost_so_far += cost
                        depth += 1
                        if is_goal_state_pool(obj_poses, goal_poses):
                            new_reward = goal_reward - cost_so_far
                            if new_reward > max_reward:
                                max_reward = new_reward
                            reward = max(max_reward * 0.5, new_reward)
                            reward = reward * (0.9 ** (depth))
                            with lock:
                                counter.value -= 1
                            return reward * simulate_goal_reward_scale
                        if is_obj_at_goal(action_goal_pose, goal_poses[obj_id]):
                            if action_type == 1:
                                obj_at_goal_list[obj_id] = one_obj_push_goal_reward
                            else:
                                obj_at_goal_list[obj_id] = one_obj_goal_reward
                        else:
                            obj_at_goal_list[obj_id] = 0
                        new_reward = np.sum(obj_at_goal_list) - cost_so_far
                        if new_reward > max_reward:
                            max_reward = new_reward
                        reward = max(max_reward * 0.5, new_reward)
                        reward = reward * (0.9 ** (depth))
                        obj_id_list = list(range(obj_num))
                        # print(f"check 8 break, Pid = {os.getpid()}")
                        break

            # do not sample this object again
            obj_id_list.remove(obj_id)
            # print(f"check 8 removed {obj_id}, Pid = {os.getpid()}")

        # print(f"check 2 for loop, Pid = {os.getpid()}")
        if depth >= max_depth:
            # reward = reward * 0.5
            # print(f"check 3 breaking out, Pid = {os.getpid()}")
            break
    # print(f"check 6, Pid = {os.getpid()}")
    with lock:
        counter.value -= 1
    return reward * simulate_goal_reward_scale


class MCTS:
    def __init__(self, time_limit: float, pool=None, motion_planner='rrt') -> None:
        self.time_limit = time_limit
        self.pool = pool
        self.motion_planner = motion_planner
        if self.pool:
            self.num_processes = self.pool._processes
        self.np_rng = np.random.default_rng(seed=42)
        random.seed(42)

        self.simulate_goal_reward_scale = 0.7

        self.boundary_box = box(BOUNDARY[0, 0], BOUNDARY[1, 0], BOUNDARY[0, 1], BOUNDARY[1, 1])

        if self.motion_planner == 'rrt':
            self.init_rrt()
        elif self.motion_planner == 'rvg':
            pass
        else:
            raise ValueError(f"Motion planner {self.motion_planner} not supported")

    def search(
        self,
        obj_polys: List[Polygon],
        obj_start_poses: np.ndarray,
        obj_goal_poses: np.ndarray,
        obj_action_types: List[int],
    ) -> None:
        """Search for the best action sequence.
        The order of the objects should not change"""

        self.goal_poses = obj_goal_poses
        self.obj_action_types = obj_action_types
        self.obj_num = len(obj_polys)
        self.max_depth = round(self.obj_num * 2) + 2
        # self.max_depth = 3
        self.max_rollout_steps = round(self.max_depth * 0.8)
        # self.one_obj_goal_reward = (self.goal_reward * 0.7) / self.obj_num
        self.one_obj_goal_reward = 0.7
        self.goal_reward = self.one_obj_goal_reward * self.obj_num * 2.0
        # self.goal_reward = self.one_obj_goal_reward * self.obj_num * 2.0
        self.one_obj_push_goal_reward = self.one_obj_goal_reward * 1.1
        self.untried_action_q = self.one_obj_goal_reward * 1.0
        self.obj_inner_radius = []
        self.obj_outer_radius = []
        self.obj_center_offset_radii = []
        self.obj_long_axis_angles = []
        for obj_poly, start_pose in zip(obj_polys, obj_start_poses):
            angle, short, long, center = compute_long_axis_radius(obj_poly)
            self.obj_outer_radius.append(long + 0.01)   # enlarge a little bit
            self.obj_inner_radius.append(short + 0.01)  # enlarge a little bit
            self.obj_center_offset_radii.append(center - start_pose[:2])
            self.obj_long_axis_angles.append(angle)
        obj_at_goal_list = are_objs_at_goal(obj_start_poses, self.goal_poses)
        print(f"Objects at goal: {obj_at_goal_list} \t max depth {self.max_depth} \t max rollout steps {self.max_rollout_steps}")
        for i in range(len(obj_at_goal_list)):
            if obj_action_types[i] == 1:
                obj_at_goal_list[i] *= self.one_obj_push_goal_reward
            else:
                obj_at_goal_list[i] *= self.one_obj_goal_reward
        self.base_reward = np.sum(obj_at_goal_list)
        # pre-compute grid actions
        self.grid_actions = []
        for radius, large_radius, center_offset, long_angle in zip(self.obj_inner_radius, self.obj_outer_radius, self.obj_center_offset_radii, self.obj_long_axis_angles):
            self.grid_actions.append(sample_grid_actions(radius, large_radius, center_offset, long_angle))
        # records
        self.depth_record = 1
        self.max_N_1 = 500
        self.max_N_2 = 3000
        self.max_N = 20000

        self.root = Node(obj_polys, obj_start_poses, self.is_goal_state(obj_start_poses))
        self.root.untried_actions = sample_actions(
            self.obj_num,
            self.obj_action_types,
            self.root.obj_poses,
            self.goal_poses,
            self.grid_actions,
            self.obj_inner_radius,
            self.obj_outer_radius,
            self.obj_center_offset_radii,
            self.obj_long_axis_angles,
        )
        # s_t = time.time()
        actions_batches = [self.root.untried_actions[i:i+3] for i in range(0, len(self.root.untried_actions), 3)]
        results = self.pool.map(
            try_actions_pool_map,
            [(actions, obj_polys, self.root.obj_poses, self.boundary_box, self.goal_poses, self.root.cost_so_far)
            for actions in actions_batches]
        )
        self.root.untried_actions = []
        for result in results:
            sub_children = result
            for child in sub_children:
                child.parent = self.root
                child.depth = self.root.depth + 1
            self.root.unexplored_children.extend(sub_children)
        self.grid_sample_half = len(self.root.unexplored_children) > 70
        if self.grid_sample_half:
            self.root.unexplored_children = self.root.unexplored_children[:50]
        # e_t = time.time()
        # print(f"init time: {e_t - s_t:.3f}")

        pbar = tqdm(total=self.time_limit, ncols=70, bar_format='{l_bar}{bar}{n:.1f}/{total:.1f} {postfix}', disable=False)
        start_time = time.time()
        prev_time = time.time()
        duration = 0
        itr = 0
        pool_results = []
        while duration < self.time_limit:
            # Selection and Expansion
            node, do_simulate = self._select_and_expand()

            # Simulation and Backpropagation
            if do_simulate:
                if self.pool:
                    self._backpropagate_virtual_visits(node)
                    with lock:
                        counter.value += 1
                    result = self.pool.apply_async(
                        simulate_pool,
                        args=(
                            node.obj_poses,
                            node.cost_so_far,
                            node.obj_polys,
                            self.obj_inner_radius,
                            self.obj_outer_radius,
                            self.obj_center_offset_radii,
                            self.obj_long_axis_angles,
                            node.depth,
                            self.goal_poses,
                            self.grid_actions,
                            self.goal_reward,
                            self.one_obj_goal_reward,
                            self.one_obj_push_goal_reward,
                            self.simulate_goal_reward_scale,
                            self.obj_num,
                            self.obj_action_types,
                            self.boundary_box,
                            self.grid_sample_half,
                            self.max_depth,
                            self.max_rollout_steps,
                            self.motion_planner
                        ),
                    )
                    pool_results.append((result, node))
                    while True:
                        for pi in reversed(range(len(pool_results))):
                            result, pool_node = pool_results[pi]
                            if result.ready():
                                reward = result.get()
                                # with lock:
                                    # print("pids = ", pids)
                                    # print("Pid removed", pid)
                                    # pids.remove(pid)
                                    # print("pids = ", pids)
                                reward = max(0, reward - self.base_reward)
                                self._backpropagate(pool_node, reward)
                                del pool_results[pi]
                        if counter.value <= self.num_processes and len(pool_results) < self.num_processes:
                        # if counter.value <= self.num_processes and len(pool_results) <= 1.5 * self.num_processes:
                        # if counter.value <= self.num_processes or len(pool_results) == 0:
                            break
                        else:  # wait for a while
                            time.sleep(0.0001)
                            # print("sim sleep", len(pool_results))
                else:
                    reward = self._simulate(node)
                    reward = max(0, reward - self.base_reward)
                    self._backpropagate(node, reward)
            else:
                if self.pool:
                    self._backpropagate_virtual_visits(node)
                reward = self.goal_reward - node.cost_so_far - self.base_reward if node.at_goal else 0
                reward = reward * (0.9 ** (node.depth))
                reward = max(0, reward)
                self._backpropagate(node, reward)

            # if the first level is fully explored and the goal is reached, stop the search
            if node.depth == 1 and self.is_goal_state(node.obj_poses):
                self.max_N = self.max_N_1
            # if no solution found in the first level, but found in the second level, stop the search
            if self.root.is_fully_expanded and node.depth == 2 and self.max_N > self.max_N_2 and self.is_goal_state(node.obj_poses):
                self.max_N = self.max_N_2
            if self.root.N > self.max_N:
                break

            curr_time = time.time()
            itr += 1
            duration = curr_time - start_time
            if duration < self.time_limit:
                pbar.update(curr_time - prev_time)
            prev_time = curr_time
            pbar.set_postfix(iterations=itr)                

        if self.pool:
            print()
            while len(pool_results) > 0:
                # print('waiting for the sim pool to finish', len(pool_results))
                # print("counter.value = ", counter.value)
                i = 0
                while i < len(pool_results):
                    result, node = pool_results[i]
                    wait_time = time.time()
                    # print("waiting", i, " Pid = ", os.getpid())
                    # print("pids = ", pids)
                    while not result.ready() and time.time() - wait_time < 1:
                        # print("sleeping", i)
                        time.sleep(0.1)
                    if result.ready():
                        reward = result.get()
                        # with lock:
                        #     pids.remove(pid)
                        reward = max(0, reward - self.base_reward)
                        self._backpropagate(node, reward)
                        del pool_results[i]
                    i+=1
            # print('waiting for the expand pool to finish', counter.value)
            while counter.value != 0:
                time.sleep(0.1)
                if time.time() - start_time > self.time_limit + 2:
                    print('timeout', counter.value)
                    counter.value = 0
                    break
        assert self.root.VN == 0
        pbar.close()

        print(f"MCTS search: {itr}, depth record: {self.depth_record}, Time: {duration:.3f}")

        # # vis all actions
        # if self.root.is_fully_expanded:
        #     best_node = self.best_node()
        #     obj_id = best_node.prev_action.obj_id
        #     obj_id = 1
        #     obj_polys = self.root.obj_polys.copy()
        #     for child in self.root.children:
        #         if child.prev_action.obj_id == obj_id:
        #             obj_polys.append(child.obj_polys[obj_id])
        #         # obj_polys.append(child.obj_polys[child.prev_action.obj_id])
        #     # obj_id = 3
        #     # obj_polys = self.root.obj_polys.copy()
        #     # for child in self.root.children:
        #     #     if child.prev_action.obj_id == obj_id:
        #     #         obj_polys.append(child.obj_polys[obj_id])
        #     from utils import plot_polygons_with_label
        #     plot_polygons_with_label(obj_polys, self.obj_action_types, f"test-sample-{itr}")
        #     exit()

    def best_node(self) -> Node:
        return self.root.best_child()

    def _select_and_expand(self) -> tuple[Node, bool]:
        """Select a node and expand it if it is not fully expanded
        Return: the expanded node and a boolean indicating whether the node should have a simulation"""
        node = self.root

        while True:
            # If the node is a leaf node, return it
            if node.is_terminal(self.max_depth):
                do_simulate = False
                break

            if not node.has_children:
                do_expand = True
            else:
                untried_score = self.untried_action_q + C_PUCT * math.sqrt(math.log(node.N + node.VN))
                child_node, score = node.best_child_with_explore(parallel=(self.pool is not None))
                # try an action that has not been tried before
                if score < untried_score and not node.is_fully_expanded:
                    do_expand = True
                else:
                    node = child_node
                    do_expand = False

            if do_expand:
                child_node, expanded = self._expand(node)
                if child_node.depth > self.depth_record:
                    self.depth_record = child_node.depth
                if expanded:
                    node = child_node
                    do_simulate = not child_node.is_terminal(self.max_depth)
                else:
                    do_simulate = False
                break

        return node, do_simulate

    def _expand(self, node: Node) -> tuple[Node, bool]:
        if node.untried_actions is None:
            node.untried_actions = sample_actions(
                self.obj_num,
                self.obj_action_types,
                node.obj_poses,
                self.goal_poses,
                self.grid_actions,
                self.obj_inner_radius,
                self.obj_outer_radius,
                self.obj_center_offset_radii,
                self.obj_long_axis_angles,
                self.grid_sample_half,
            )
        # cutoff, limit the branching factor, TODO: remove this if performance is not good
        # if len(node.children) > 150:
        #     node.untried_actions = []
        #     node.unexplored_children = []
        #     node.pool_results = []
        # if isinstance(node.untried_actions, multiprocessing.pool.ApplyResult):
        #     node.untried_actions = node.untried_actions.get()
        if self.pool:
            # check if we can get a result from the pool
            if len(node.unexplored_children) == 0 and len(node.pool_results) > 0:
                for pi in reversed(range(len(node.pool_results))):
                    if node.pool_results[pi].ready():
                        sub_children = node.pool_results[pi].get()
                        for child in sub_children:
                            child.parent = node
                            child.depth = node.depth + 1
                        node.unexplored_children.extend(sub_children)
                        del node.pool_results[pi]
                        break
            # pop one child node from the unexplored children
            if len(node.unexplored_children) > 0:
                child_node = node.unexplored_children.pop(0)
                node.children.append(child_node)
                return child_node, True
            # prepare children
            else:
                obj_polys = node.obj_polys.copy()
                obj_poses = node.obj_poses.copy()
                while not node.is_fully_expanded and len(node.unexplored_children) == 0:
                    if counter.value > self.num_processes:
                        time.sleep(0.0001)
                        # print("expand sleep")
                    else:
                        actions = node.untried_actions[0:3]
                        if len(actions) > 0:
                            del node.untried_actions[0:3]
                            with lock:
                                counter.value += 1
                            result = self.pool.apply_async(
                                try_actions_pool,
                                args=(
                                    actions,
                                    obj_polys,
                                    obj_poses,
                                    self.boundary_box,
                                    self.goal_poses,
                                    node.cost_so_far,
                                ),
                            )
                            node.pool_results.append(result)
                    pi = 0
                    while pi < len(node.pool_results):
                        if node.pool_results[pi].ready():
                            sub_children = node.pool_results[pi].get()
                            for child in sub_children:
                                child.parent = node
                                child.depth = node.depth + 1
                            node.unexplored_children.extend(sub_children)
                            del node.pool_results[pi]
                            break
                        else:
                            pi += 1
                if len(node.unexplored_children) > 0:
                    child_node = node.unexplored_children.pop(0)
                    # TODO: prm multi-query?
                    node.children.append(child_node)
                    return child_node, True
        else:
            obj_polys = node.obj_polys.copy()
            obj_poses = node.obj_poses.copy()
            is_valid_action = False

            # select an action from the untried actions
            while not node.is_fully_expanded:
                action = node.untried_actions.pop(0)

                # check if the action is valid
                obj_poly = obj_polys[action.obj_id]
                obj_pose = obj_poses[action.obj_id]
                if is_obj_at_goal(obj_pose, action.goal_pose):
                    continue
                obs_polys = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
                move = action.goal_pose - obj_pose
                new_obj_poly = shapely_rotate_translate_with_center(
                    obj_poly, move[0], move[1], move[2], obj_pose[0], obj_pose[1]
                )
                if new_obj_poly.within(self.boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
                    if action.type == 0:
                        cost = get_pick_place_cost(obj_pose, action.goal_pose)
                        is_valid_action = True
                    else:
                        solved, cost, _ = self.solve_drag_rrt(obj_polys, obj_poses, action)
                        if solved:
                            is_valid_action = True

                    if is_valid_action:
                        obj_polys[action.obj_id] = new_obj_poly
                        obj_poses[action.obj_id] = action.goal_pose
                        child_node = Node(
                            obj_polys,
                            obj_poses,
                            self.is_goal_state(obj_poses),
                            node,
                            action,
                            cost_so_far=node.cost_so_far + cost,
                        )
                        node.children.append(child_node)
                        return child_node, True

        return node, False

    def _simulate(self, node: Node) -> float:
        """Simulate the environment and return the reward"""
        if self.is_goal_state(node.obj_poses):
            return self.goal_reward - node.cost_so_far - self.base_reward

        obj_polys = node.obj_polys.copy()
        obj_poses = node.obj_poses.copy()
        cost_so_far = node.cost_so_far
        depth = node.depth
        obj_id_list = list(range(self.obj_num))
        obj_at_goal_list = are_objs_at_goal(obj_poses, self.goal_poses)
        for i in range(len(obj_at_goal_list)):
            if self.obj_action_types[i] == 1:
                obj_at_goal_list[i] *= self.one_obj_push_goal_reward
            else:
                obj_at_goal_list[i] *= self.one_obj_goal_reward
        reward = np.sum(obj_at_goal_list) - cost_so_far - self.base_reward
        reward = max(0, reward)

        for _ in range(self.max_rollout_steps):
            if len(obj_id_list) == 0:
                break
            # sample an object to move
            obj_id = random.choice(obj_id_list)
            curr_pose = obj_poses[obj_id]
            goal_pose = self.goal_poses[obj_id]
            action_type = self.obj_action_types[obj_id]
            obj_poly = obj_polys[obj_id]
            obs_polys = obj_polys[:obj_id] + obj_polys[obj_id + 1 :]

            # select an action
            is_valid_action = False
            action_goal_poses = list(
                sample_next_poses(curr_pose, goal_pose, self.grid_actions[obj_id], self.obj_inner_radius[obj_id], self.obj_outer_radius[obj_id], is_simulate=True)
            )
            # piroritize the goal pose
            if random.random() < 0.2:
                if not is_obj_at_goal(curr_pose, goal_pose):
                    action_goal_poses = [goal_pose] + action_goal_poses
                random.shuffle(action_goal_poses)
            else:
                random.shuffle(action_goal_poses)
                if not is_obj_at_goal(curr_pose, goal_pose):
                    action_goal_poses = [goal_pose] + action_goal_poses
            for action_goal_pose in action_goal_poses:
                if is_obj_at_goal(curr_pose, action_goal_pose):
                    continue
                # check if the action is valid
                move = action_goal_pose - curr_pose
                new_obj_poly = shapely_rotate_translate_with_center(
                    obj_poly, move[0], move[1], move[2], curr_pose[0], curr_pose[1]
                )
                if new_obj_poly.within(self.boundary_box) and not shapely_collision_check(new_obj_poly, obs_polys):
                    if action_type == 0:
                        cost = get_pick_place_cost(curr_pose, action_goal_pose)
                        is_valid_action = True
                    else:
                        solved, cost, path = self.solve_drag_rrt(
                            obj_polys, obj_poses, Action(action_type, obj_id, action_goal_pose)
                        )
                        if solved:
                            is_valid_action = True

                    # execute the action
                    if is_valid_action:
                        obj_polys[obj_id] = new_obj_poly
                        obj_poses[obj_id] = action_goal_pose
                        cost_so_far += cost
                        depth += 1
                        if self.is_goal_state(obj_poses):
                            new_reward = self.goal_reward - cost_so_far - self.base_reward
                            reward = max(reward, new_reward)
                            return reward * self.simulate_goal_reward_scale
                        if is_obj_at_goal(action_goal_pose, self.goal_poses[obj_id]):
                            if action_type == 1:
                                obj_at_goal_list[obj_id] = self.one_obj_push_goal_reward
                            else:
                                obj_at_goal_list[obj_id] = self.one_obj_goal_reward
                        else:
                            obj_at_goal_list[obj_id] = 0
                        new_reward = np.sum(obj_at_goal_list) - cost_so_far - self.base_reward
                        reward = max(reward, new_reward)
                        obj_id_list = list(range(self.obj_num))
                        break

            # do not sample this object again
            obj_id_list.remove(obj_id)

            if depth >= self.max_depth:
                reward = reward * 0.1
                break

        return reward * self.simulate_goal_reward_scale

    def _backpropagate(self, node: Node, reward: float) -> None:
        node.N += 1
        if self.pool:
            node.VN -= 1
            assert node.VN >= 0
        # node.Q += reward
        node.Q.push(reward)
        if node.parent is not None:
            self._backpropagate(node.parent, reward)

    def _backpropagate_virtual_visits(self, node: Node):
        node.VN += 1
        if node.parent is not None:
            self._backpropagate_virtual_visits(node.parent)

    def init_rrt(self) -> None:
        space = CustomSE2StateSpace()
        bounds = ob.RealVectorBounds(2)  # type: ignore
        bounds.setLow(0, BOUNDARY[0, 0])
        bounds.setHigh(0, BOUNDARY[0, 1])
        bounds.setLow(1, BOUNDARY[1, 0])
        bounds.setHigh(1, BOUNDARY[1, 1])
        bounds.setLow(2, -math.pi)
        bounds.setHigh(2, math.pi)
        space.setBounds(bounds)

        self.rrt_ss = og.SimpleSetup(space)
        self.rrt_ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(MCTS.is_state_valid_rrt, self)))
        space.setup()
        self.rrt_ss.getSpaceInformation().setStateValidityCheckingResolution(0.01)

        # RRTConnect, LazyLBTRRT, InformedRRTstar, SORRTstar
        self.planner_fast = og.RRTConnect(self.rrt_ss.getSpaceInformation())
        self.planner_fast.setRange(0.3)
        self.planner_fast.setup()

        self.planner_optimal = og.LazyLBTRRT(self.rrt_ss.getSpaceInformation())
        self.planner_optimal.setRange(0.3)
        self.planner_optimal.setup()

    def is_state_valid_rrt(self, state) -> bool:
        pose = (state.getX(), state.getY(), state.getYaw())
        robot_shape = shapely_rotate_translate_with_center(self.rrt_obj_poly, pose[0], pose[1], pose[2], 0, 0)

        if not robot_shape.within(self.boundary_box):
            return False

        if robot_shape.intersects(self.rrt_obs_polys):
            return False

        return True

    def solve_drag_rrt(self, obj_polys: List[Polygon], obj_poses: np.ndarray, action: Action, optimal=False):
        """Check if the goal pose is reachable"""

        obj_poly = obj_polys[action.obj_id]
        obj_pose = obj_poses[action.obj_id].copy()
        obstacles = obj_polys[: action.obj_id] + obj_polys[action.obj_id + 1 :]
        self.rrt_obs_polys = unary_union(obstacles)
        goal_pose = action.goal_pose

        # rotate and translate the object to the origin
        self.rrt_obj_poly = shapely_rotate_translate_with_center(
            obj_poly, -obj_pose[0], -obj_pose[1], -obj_pose[2], obj_pose[0], obj_pose[1]
        )

        if self.motion_planner == 'rrt':
            self.rrt_ss.clear()
            start = ob.State(self.rrt_ss.getStateSpace())
            start().setX(obj_pose[0])
            start().setY(obj_pose[1])
            start().setYaw(obj_pose[2])
            goal = ob.State(self.rrt_ss.getStateSpace())
            goal().setX(goal_pose[0])
            goal().setY(goal_pose[1])
            goal().setYaw(goal_pose[2])
            self.rrt_ss.setStartAndGoalStates(start, goal)
            if not optimal:
                self.rrt_ss.setPlanner(self.planner_fast)
            else:
                self.rrt_ss.setPlanner(self.planner_optimal)

            if not optimal:
                try:
                    # signal.alarm(1)
                    self.rrt_ss.solve(0.2)
                    # signal.alarm(0)
                except TimeoutError:
                    print("timeout, try again")
                    self.rrt_ss.clear()
                # finally:
                #     signal.alarm(0)
                if self.rrt_ss.haveExactSolutionPath():
                    path = self.rrt_ss.getSolutionPath()
                    cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                    return True, cost, path
                else:
                    return False, None, None
            else:
                t = 2
                count = 0
                while True:
                    try:
                        signal.alarm(t)
                        self.rrt_ss.solve(t - 0.1)
                        signal.alarm(0)
                        if self.rrt_ss.haveExactSolutionPath():
                            break
                        else:
                            t = 3
                            print("no exact solution, try again")
                            self.rrt_ss.clear()
                    except TimeoutError:
                        t = 3
                        print("timeout, try again")
                        self.rrt_ss.clear()
                    finally:
                        signal.alarm(0)
                    count += 1
                    if count > 3:
                        print('switch to fast planner')
                        self.rrt_ss.setPlanner(self.planner_fast)

                # ns = self.rrt_ss.getProblemDefinition().getSolutionCount()
                # print("Found %d solutions" % ns)
                signal.alarm(t)
                if self.rrt_ss.haveExactSolutionPath():
                    path = self.rrt_ss.getSolutionPath()
                    ps = og.PathSimplifier(self.rrt_ss.getSpaceInformation())
                    ps.simplifyMax(path)
                    if len(path.getStates()) > 2:
                        ps.smoothBSpline(path, 2)
                    else:
                        path.interpolate(int(path.length() / 0.1))
                        # ps.smoothBSpline(path, 4)
                    cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * path.length()
                    signal.alarm(0)
                    return True, cost, path
                else:
                    signal.alarm(0)
                    return False, None, None

        elif self.motion_planner == 'rvg': 
            obj = shapely_polygon_to_rvg_polygon(self.rrt_obj_poly)
            boundary = rvg.polygon([
                rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 0]),
                rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 0]),
                rvg.vertex(BOUNDARY[0, 1], BOUNDARY[1, 1]),
                rvg.vertex(BOUNDARY[0, 0], BOUNDARY[1, 1]),
                ], False)
            obstacles = [
                shapely_polygon_to_rvg_polygon(obs)
                for obs in obstacles
            ]
            start = rvg.vertex(obj_pose[0], obj_pose[1], 0, 2 * np.pi, obj_pose[2], 2 * np.pi, True)
            goal = rvg.vertex(goal_pose[0], goal_pose[1], 0, 2 * np.pi, goal_pose[2], 2 * np.pi, True)

            vg = rvg.visibility_graph(robot=obj, border = boundary, obstacles = obstacles, resolution=36, considerSymmetry=True, hashWithTheta=True, simplifiedGeometry=False, numThreads=1, incremental=False, verbose=False)
            vg.setWeight(0.5, 0.5)
            path = vg.shortestPath(start, goal)
            if len(path) == 0:
                return False, None, None
            else:
                cost = PICK_DRAG_BASE_COST + PICK_DRAG_DIST_SCALE * vg.getPathLength() 
                return True, cost, path
        else:
            raise ValueError(f"Motion planner {self.motion_planner} not supported")


    def is_state_equal(self, poses1: np.ndarray, poses2: np.ndarray) -> bool:
        diff = poses1[:, 2] - poses2[:, 2]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return (
            np.allclose(poses1[:, 0:2], poses2[:, 0:2], atol=IS_CLOSE_TRANS_THRESHOLD)
            and np.all(np.abs(diff) < IS_CLOSE_ROT_THRESHOLD)
        )

    def is_goal_state(self, poses: np.ndarray) -> bool:
        return self.is_state_equal(poses, self.goal_poses)
