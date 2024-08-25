import argparse
import atexit
import multiprocessing
import time
import ast
from functools import partial
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
import pybullet as pb
from pybullet_utils import bullet_client
import pybullet_data
from shapely.geometry import Polygon
from shapely.affinity import scale

from utils import (
    plot_polygons_with_drag,
    plot_polygons,
    shapely_rotate_with_center,
    shapely_translate,
    shapely_rotate_translate_with_center,
    extrude_polygon,
    plot_polygons_with_label
)
from mcts.node import Action
from mcts.search import MCTS
from greedy import Greedy_Solver
from constants import BOUNDARY

class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui

        # self.home_joints = np.array([67.36, -80.58, 12.81, -22.13, -89.82, -112.01])
        # for i in range(len(self.home_joints)):
        #     self.home_joints[i] = np.deg2rad(self.home_joints[i])
        self.home_joints = np.array([1.57, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([1.57, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.obj_height = 0.05

        # Start PyBullet.
        self._pb = bullet_client.BulletClient(connection_mode=pb.GUI if gui else pb.DIRECT)
        self._client_id = self._pb._client
        self._pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pb.setTimeStep(time_step)

        if gui:
            target = self._pb.getDebugVisualizerCamera()[11]
            self._pb.resetDebugVisualizerCamera(
                cameraDistance=1.2, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=target,
            )

    def reset(self):
        self._pb.resetSimulation()
        self._pb.setGravity(0, 0, -9.8)

        if self.gui:
            self._pb.configureDebugVisualizer(self._pb.COV_ENABLE_GUI, 0)

        # Load workspace
        self.plane = self._pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = self._pb.loadURDF(
            "assets/workspace.urdf", basePosition=(0., -0.5, 0), useFixedBase=True,
        )
        self._pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.0,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        self._pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.0,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = self._pb.loadURDF(
            "assets/ur5e/ur5e_suction.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(self._pb.getNumJoints(self.ur5e)):
            info = self._pb.getJointInfo(self.ur5e, i)
            joint_link_id = info[0]
            link_name = info[12].decode("utf-8")
            joint_type = info[2]
            if link_name == 'tipLink':
                self.ee_tip = joint_link_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_link_id)

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False
        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        self.go_home()

    def draw_boundary(self):
        line_color = [1, 0, 0]
        line_width = 3
        height = 0.01
        self._pb.addUserDebugLine([BOUNDARY[0, 0], BOUNDARY[1, 0], height], [BOUNDARY[0, 0], BOUNDARY[1, 1], height], line_color, line_width, physicsClientId=self._client_id)
        self._pb.addUserDebugLine([BOUNDARY[0, 0], BOUNDARY[1, 0], height], [BOUNDARY[0, 1], BOUNDARY[1, 0], height], line_color, line_width, physicsClientId=self._client_id)
        self._pb.addUserDebugLine([BOUNDARY[0, 1], BOUNDARY[1, 1], height], [BOUNDARY[0, 0], BOUNDARY[1, 1], height], line_color, line_width, physicsClientId=self._client_id)
        self._pb.addUserDebugLine([BOUNDARY[0, 1], BOUNDARY[1, 1], height], [BOUNDARY[0, 1], BOUNDARY[1, 0], height], line_color, line_width, physicsClientId=self._client_id)


    def add_objects(self, obj_poses, obj_polys):
        """Adds objects to the environment.

        Args:
        obj_poses: list of object poses, each pose is a list of [x, y, theta]
        obj_polys: list of object polygons, each polygon is a list of [x, y]
        """

        color_space = (
            np.asarray(
                [
                    [78.0, 121.0, 167.0],  # blue
                    [89.0, 161.0, 79.0],  # green
                    [156, 117, 95],  # brown
                    [242, 142, 43],  # orange
                    [237.0, 201.0, 72.0],  # yellow
                    [186, 176, 172],  # gray
                    [255.0, 87.0, 89.0],  # red
                    [176, 122, 161],  # purple
                    [118, 183, 178],  # cyan
                    [255, 157, 167],  # pink
                ]
            )
            / 255.0
        )

        self.obj_ids = []
        for i, (obj_pose, obj_poly) in enumerate(zip(obj_poses, obj_polys)):
            obj_file_name = f"logs/temp-{i}.obj"
            extrude_polygon(obj_poly, self.obj_height, obj_file_name)
            obj_id = self._pb.createMultiBody(
                baseMass=0.001,
                baseCollisionShapeIndex=self._pb.createCollisionShape(
                    shapeType=self._pb.GEOM_MESH,
                    fileName=obj_file_name,
                ),
                basePosition=obj_pose[:2] + [0.025],
                baseOrientation=self._pb.getQuaternionFromEuler([0, 0, obj_pose[2]]),
                useMaximalCoordinates=True,
                physicsClientId=self._client_id
            )
            self._pb.changeDynamics(
                obj_id,
                -1,
                lateralFriction=1.0,
                rollingFriction=0.1,
                spinningFriction=0.1,
            )

            object_color = [*color_space[i], 1]
            self._pb.changeVisualShape(obj_id, -1, rgbaColor=object_color)
            self.obj_ids.append(obj_id)
            self._pb.stepSimulation(physicsClientId=self._client_id)

    def get_obj_poses(self, obj_id):
        return self._pb.getBasePositionAndOrientation(self.obj_ids[obj_id], physicsClientId=self._client_id)

    def get_link_pose(self, body, link):
        result = self._pb.getLinkState(body, link)
        return result[0], result[1]

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=10):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    self._pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ur5e, self.ee_tip)
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 7e-3):
                # give time to stop
                for _ in range(5):
                    self._pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            self._pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            self._pb.stepSimulation()

        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.0005):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = self._pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283 * 2],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283 * 2],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566 * 2],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.0001, detect_contact=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.002  # every 0.5 cm
        vec = pose1 - pose0
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance)) 
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_contact and self.detect_contact():
                success = True
                break

        # success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def suction_activate(self) -> bool:
        """Simulate suction using a rigid fixed constraint to contacted object."""
        if not self.activated:
            points = self._pb.getContactPoints(bodyA=self.ur5e, linkIndexA=self.ee_tip)
            if points:
                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids:
                    body_pose = self._pb.getLinkState(self.ur5e, self.ee_tip)
                    obj_pose = self._pb.getBasePositionAndOrientation(obj_id)
                    world_to_body = self._pb.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = self._pb.multiplyTransforms(world_to_body[0],
                                                        world_to_body[1],
                                                        obj_pose[0], obj_pose[1])

                    self.contact_constraint = self._pb.createConstraint(
                        parentBodyUniqueId=self.ur5e,
                        parentLinkIndex=self.ee_tip,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=self._pb.JOINT_FIXED,
                        jointAxis=(0, 0, 1),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0, 1),
                    )
                    self.activated = True
                    return True
        return False

    def suction_release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False

        # Release gripped rigid object (if any).
        if self.contact_constraint is not None:
            try:
                self._pb.removeConstraint(self.contact_constraint)
                self.contact_constraint = None
            except:
                pass

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = self._pb.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None            

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.ur5e, self.ee_tip
        if self.activated and self.contact_constraint is not None:
            try:
                info = self._pb.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = self._pb.getContactPoints(bodyA=body, linkIndexA=link)

        if self.activated:
            points = [point for point in points if point[2] != self.ur5e]

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def pick_place(self, pick_pose, place_pose) -> bool:
        pre_pick = pick_pose.copy()
        pre_pick[2] += 0.1
        after_pick = pick_pose.copy()
        after_pick[2] += 0.15
        pre_place = place_pose.copy()
        pre_place[2] += 0.1

        # Move to pre_pick pose
        success = self.move_ee_pose((pre_pick[0:3], self._pb.getQuaternionFromEuler(pre_pick[3:6])))
        # Move down to pick pose
        if success:
            success = self.straight_move(pre_pick[0:3], pick_pose[0:3], self._pb.getQuaternionFromEuler(pick_pose[3:6]), detect_contact=True)
        # Activate suction
        if success:
            success = self.suction_activate()
        # Move up to after_pick pose
        if success:
            success = self.move_ee_pose((after_pick[0:3], self._pb.getQuaternionFromEuler(after_pick[3:6])))
        # Move to pre_place pose
        if success:
            success = self.move_ee_pose((pre_place[0:3], self._pb.getQuaternionFromEuler(pre_place[3:6])))
        # Move down to place pose
        if success:
            success = self.straight_move(pre_place[0:3], place_pose[0:3], self._pb.getQuaternionFromEuler(place_pose[3:6]), detect_contact=True)
        # Release suction and go home
        self.suction_release()
        # Go up a bit
        if success:
            pos, rot = self.get_link_pose(self.ur5e, self.ee_tip)
            pos = np.array(pos)
            pos_up = pos.copy()
            pos_up[2] += 0.05
            success = self.straight_move(pos, pos_up, rot, detect_contact=False)
        self.go_home()

        return success

    def pick_drag(self, move_poses) -> bool:
        pre_pick = move_poses[0].copy()
        pre_pick[2] += 0.1
        after_pick = move_poses[0].copy()
        after_pick[2] += 0.05

        # Move to pre_pick pose
        success = self.move_ee_pose((pre_pick[0:3], self._pb.getQuaternionFromEuler(pre_pick[3:6])))
        # Move down to pick pose
        if success:
            success = self.straight_move(pre_pick[0:3], move_poses[0][0:3], self._pb.getQuaternionFromEuler(move_poses[0][3:6]), detect_contact=True)
        # Activate suction
        if success:
            success = self.suction_activate()
        # Move up to after_pick pose
        if success:
            success = self.move_ee_pose((after_pick[0:3], self._pb.getQuaternionFromEuler(after_pick[3:6])))
        # follow the path
        if success:
            # for move in move_poses[1:]:
            #     move[2] += 0.05
            #     success = self.move_ee_pose((move[0:3], self._pb.getQuaternionFromEuler(move[3:6])))
            #     if not success:
            #         break
            for i in range(len(move_poses) - 1):
                start = move_poses[i].copy()
                end = move_poses[i + 1].copy()
                start[2] += 0.05
                end[2] += 0.05
                success = self.straight_move(start[0:3], end[0:3], self._pb.getQuaternionFromEuler(end[3:6]), detect_contact=False)
                if not success:
                    break
        # Release suction and go home
        self.suction_release()
        # Go up a bit
        if success:
            pos, rot = self.get_link_pose(self.ur5e, self.ee_tip)
            pos = np.array(pos)
            pos_up = pos.copy()
            pos_up[2] += 0.05
            success = self.straight_move(pos, pos_up, rot, detect_contact=False)
        self.go_home()

        return success

def prepare_case(case_name: str):
    with open(f"sim_tests/{case_name}.txt", "r") as f:
        contents = f.read()
    obj_poses_str = contents.split("obj_poses = ")[1].split("\n")[0]
    obj_goal_poses_str = contents.split("obj_goal_poses = ")[1].split("\n")[0]
    obj_polys_str = contents.split("obj_polys = ")[1].split("\n")[0]
    action_types_str = contents.split("action_types = ")[1].split("\n")[0]

    obj_start_poses = ast.literal_eval(obj_poses_str)
    obj_goal_poses = ast.literal_eval(obj_goal_poses_str)
    obj_polys = ast.literal_eval(obj_polys_str)
    obj_action_types = ast.literal_eval(action_types_str)

    return obj_start_poses, obj_goal_poses, obj_polys, obj_action_types


def clean_pool(pool):
    pool.terminate()
    pool.close()
    pool.join()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", action="store", choices=["mcts", "greedy"], help="search method to use")
    parser.add_argument("--motion_planner", action="store", choices=["rrt", "rvg"], default='rrt', help="search method to use")
    parser.add_argument("--case", action="store", required=True, type=str, help="case to run")
    parser.add_argument("--log", action="store", required=False, type=str, help="where to log")
    parser.add_argument("--gui", action="store_true", required=False, help="show gui")
    args = parser.parse_args()
    search_method = args.method
    case_name = args.case
    motion_planner = args.motion_planner
    if args.gui:
        gui = True
    else:
        gui = False


    if search_method == "mcts":
        pool = multiprocessing.Pool(processes=1)
        print(f"case {case_name}, multiprocessing {pool._processes} processes are used! Using motion planner {motion_planner}")
        atexit.register(partial(clean_pool, pool))
        solver = MCTS(5, pool, motion_planner=motion_planner)
    elif search_method == "greedy":
        solver = Greedy_Solver()

    # load case
    obj_start_poses, obj_goal_poses, obj_polys_list, obj_action_types = prepare_case(case_name)

    # prepare case in env
    env = Environment(gui=gui)
    env.reset()
    env.add_objects(obj_start_poses, obj_polys_list)
    env.draw_boundary()
    env.go_home()

    # plan
    obj_polys = []
    obj_polys_at_goal = []
    for poly, pose, goal_pose in zip(obj_polys_list, obj_start_poses, obj_goal_poses):
        poly = Polygon(poly)
        poly = poly.buffer(0.01, join_style=2)  # enlarge the polygon a bit, make the drag easier
        poly_goal = shapely_rotate_translate_with_center(
            poly, goal_pose[0], goal_pose[1], goal_pose[2], 0, 0
        )
        poly = shapely_rotate_translate_with_center(
            poly, pose[0], pose[1], pose[2], 0, 0
        )
        obj_polys.append(poly)
        obj_polys_at_goal.append(poly_goal)
    obj_start_poses = np.array(obj_start_poses)
    obj_goal_poses = np.array(obj_goal_poses)
    actions = []
    step = 0
    planning_time = 0
    robot_time = 0
    total_cost = 0
    while not solver.is_state_equal(obj_start_poses, obj_goal_poses) and step < 20:
        s_t = time.time()
        if search_method == "mcts":
            solver.search(copy.deepcopy(obj_polys), obj_start_poses, obj_goal_poses, obj_action_types)
            node = solver.best_node()

            action = node.prev_action
            obj_id = action.obj_id
            xoff, yoff, angle = (
                action.goal_pose[0] - obj_start_poses[obj_id][0],
                action.goal_pose[1] - obj_start_poses[obj_id][1],
                action.goal_pose[2] - obj_start_poses[obj_id][2],
            )
            total_cost += node.cost_so_far
        elif search_method == "greedy":
            obj_id, goal_pose, cost = solver.search(obj_polys, obj_start_poses, obj_goal_poses, obj_action_types)
            action = Action(obj_action_types[obj_id], obj_id, goal_pose)
            xoff, yoff, angle = (
                goal_pose[0] - obj_start_poses[obj_id][0],
                goal_pose[1] - obj_start_poses[obj_id][1],
                goal_pose[2] - obj_start_poses[obj_id][2],
            )
            total_cost += cost

        prev_obj_start_poses = obj_start_poses[obj_id].copy()
        if action.type == 0:
            e_t = time.time()
            obj_polys[obj_id] = shapely_rotate_translate_with_center(
                obj_polys[obj_id], xoff, yoff, angle, obj_start_poses[obj_id][0], obj_start_poses[obj_id][1]
            )
            obj_start_poses[obj_id] = [action.goal_pose[0], action.goal_pose[1], action.goal_pose[2]]
            plot_polygons(obj_polys, f"logs/step-{step}")
            act = [action.type, obj_id, prev_obj_start_poses.copy(), obj_start_poses[obj_id].copy()]
            actions.append(act)
        else:
            solved, cost, path = solver.solve_drag_rrt(obj_polys, obj_start_poses, action, optimal=True)
            e_t = time.time()
            assert solved
            poses = []
            if motion_planner == "rvg":
                for state in path:
                    poses.append((state.getX(), state.getY(), state.getTheta()))
            elif motion_planner == "rrt":
                for state in path.getStates():
                    poses.append((state.getX(), state.getY(), state.getYaw()))
            ori_poly = shapely_rotate_translate_with_center(
                obj_polys[obj_id],
                -obj_start_poses[obj_id][0],
                -obj_start_poses[obj_id][1],
                -obj_start_poses[obj_id][2],
                obj_start_poses[obj_id][0],
                obj_start_poses[obj_id][1],
            )
            plot_polygons_with_drag(obj_polys, ori_poly, poses, f"logs/step-{step}")
            obj_polys[obj_id] = shapely_rotate_translate_with_center(
                obj_polys[obj_id], xoff, yoff, angle, obj_start_poses[obj_id][0], obj_start_poses[obj_id][1]
            )
            obj_start_poses[obj_id] = [action.goal_pose[0], action.goal_pose[1], action.goal_pose[2]]
            act = [action.type, obj_id, poses]
            actions.append(act)
        planning_time += (e_t - s_t)

        # robot sim execution
        s_t = time.time()
        if act[0] == 0:
            start_pose = np.array([act[2][0], act[2][1], env.obj_height / 2 + 0.005, 3.14, 0, 0])
            place_angle = act[3][2] - act[2][2]
            place_angle = (place_angle + np.pi) % (2 * np.pi) - np.pi
            goal_pose = np.array([act[3][0], act[3][1], env.obj_height / 2 + 0.005, 3.14, 0, place_angle])
            env.pick_place(start_pose, goal_pose)
            # print(start_pose, goal_pose)
        elif act[0] == 1:
            path = [np.array([act[2][0][0], act[2][0][1], env.obj_height / 2, 3.14, 0, 0])]
            for i in range(1, len(act[2])):
                path.append(np.array([act[2][i][0], act[2][i][1], env.obj_height / 2, 3.14, 0, act[2][i][2] - act[2][0][2]]))
            env.pick_drag(path)
            # print(path)
        e_t = time.time()
        robot_time += (e_t - s_t)

        print(f"Step: {step}, Action: {action.type}, Planning time: {planning_time:.3f}, Robot time: {robot_time:.3f}, Total planning cost: {total_cost:.3f}")
        step += 1
    
        # update obj_start_poses
        for i in range(len(obj_polys_list)):   
            pos, rot = env.get_obj_poses(i)
            rot = env._pb.getEulerFromQuaternion(rot)
            obj_start_poses[i][0:2] = list(pos)[0:2]
            obj_start_poses[i][2] = rot[2]
            poly = Polygon(obj_polys_list[i])
            poly = poly.buffer(0.004, join_style=2)
            obj_polys[i] = shapely_rotate_translate_with_center(
                poly, obj_start_poses[i][0], obj_start_poses[i][1], obj_start_poses[i][2], 0, 0
            )

    success = solver.is_state_equal(obj_start_poses, obj_goal_poses)
    print(f"=====Problem solved: {success}=====")
    if args.log:
        with open(args.log, "a") as file:
            file.write(f"{case_name}, {search_method}, {success}, {step}, {planning_time:.3f}, {robot_time:.3f}\n")

    plt.close("all")
