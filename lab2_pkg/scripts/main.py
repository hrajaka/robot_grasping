#!/home/cc/ee106b/sp19/class/ee106b-aai/virtualenvironment/my_new_app/bin/python


# #!/home/cc/ee106b/sp19/class/ee106b-abj/python-virtual-environments/env/bin/python


"""
Starter script for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
import scipy
import sys
import argparse
# AutoLab imports
from autolab_core import RigidTransform
import trimesh
import warnings
warnings.simplefilter("ignore", DeprecationWarning)



# 106B lab imports
from lab2.policies import GraspingPolicy
try:
    import rospy
    import tf
    from baxter_interface import gripper as baxter_gripper
    from path_planner import PathPlanner
    ros_enabled = True
    from geometry_msgs.msg import PoseStamped
except:
    print 'Couldn\'t import ROS.  I assume you\'re running this on your laptop'
    ros_enabled = False

def RigidTransformToPoseStamped(G):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "gripper_pose"

        translation = G.translation

        pose.pose.position.x = translation[0]
        pose.pose.position.y = translation[1]
        pose.pose.position.z = translation[2]

        quaternion = G.quaternion
        pose.pose.orientation.w = quaternion[0]
        pose.pose.orientation.x = quaternion[1]
        pose.pose.orientation.y = quaternion[2]
        pose.pose.orientation.z = quaternion[3]

        return pose

def matrixToPoseStamped(mat):

        rot_mat = mat[:3, :3]
        trans = mat[:3,3]

        rigid_transform = RigidTransform(rot_mat, trans, from_frame='from', to_frame='to')

        pose = RigidTransformToPoseStamped(rigid_transform)

        return pose

def lookup_transform(to_frame, from_frame='base'):
    """
    Returns the AR tag position in world coordinates

    Parameters
    ----------
    to_frame : string
        examples are: ar_marker_7, gearbox, pawn, ar_marker_3, etc
    from_frame : string
        lets be real, you're probably only going to use 'base'

    Returns
    -------
    :obj:`autolab_core.RigidTransform` AR tag position or object in world coordinates
    """
    tag_rot = None
    tag_pos = None

    print('CALLING lookup_transform')
    print('to_frame: {}, from_frame: {}'.format(to_frame, from_frame))
    if not ros_enabled:
        print 'I am the lookup transform function!  ' \
            + 'You\'re not using ROS, so I\'m returning the Identity Matrix.'
        return RigidTransform(to_frame=from_frame, from_frame=to_frame)
    print('initializing transformlistener')
    listener = tf.TransformListener()
    attempts, max_attempts, rate = 0, 10, rospy.Rate(1.0)
    while attempts < max_attempts:
        print('attempt {}'.format(attempts))
        try:
            t = listener.getLatestCommonTime(from_frame, to_frame)
            tag_pos, tag_rot = listener.lookupTransform(from_frame, to_frame, t)
        except Exception as e:
            print(e)
            rate.sleep()
        attempts += 1
    tag_rot = np.array([tag_rot[3], tag_rot[0], tag_rot[1], tag_rot[2]])
    rot = RigidTransform.rotation_from_quaternion(tag_rot)
    return RigidTransform(rot, tag_pos, to_frame=to_frame, from_frame=from_frame)

def execute_grasp(T_world_grasp, planner, gripper):
    """
    takes in the desired hand position relative to the object, finds the desired
    hand position in world coordinates.  Then moves the gripper from its starting
    orientation to some distance BEHIND the object, then move to the  hand pose
    in world coordinates, closes the gripper, then moves up.

    Parameters
    ----------
    T_grasp_world : :obj:`autolab_core.RigidTransform`
        desired position of gripper relative to the world frame
    """

    def close_gripper():
        """closes the gripper"""
        gripper.close(block=True)
        rospy.sleep(1.0)

    def open_gripper():
        """opens the gripper"""
        gripper.open(block=True)
        rospy.sleep(1.0)

    inp = raw_input('Press <Enter> to move, or \'exit\' to exit')
    if inp == "exit":
        return

    ## opening gripper ##
    print('---------- OPENING GRIPPER ----------')
    open_gripper()

    print('---------- MOVING TO INTERMEDIATE POSITION ----------')
    temp_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, -0.1],
                            [0, 0, 0, 1]])
    T1 = np.matmul(T_world_grasp.matrix, temp_matrix)
    pose_stamped_1 = matrixToPoseStamped(T1)

    plan_1 = planner.plan_to_pose(pose_stamped_1)

    planner.execute_plan(plan_1)

    print('---------- MOVING TO FINAL POSITION ----------')
    pose_stamped_2 = RigidTransformToPoseStamped(T_world_grasp)

    plan_2 = planner.plan_to_pose(pose_stamped_2)

    planner.execute_plan(plan_2)

    print('----------CLOSING GRIPPER----------')
    close_gripper()

    print('---------- MOVING TO UP POSITION ----------')
    pose_stamped_3 = RigidTransformToPoseStamped(T_world_grasp)
    pose_stamped_3.pose.position.z = pose_stamped_3.pose.position.z + 0.3

    plan_3 = planner.plan_to_pose(pose_stamped_3)

    planner.execute_plan(plan_3)

    print('---------- MOVING TO LATERAL POSITION ----------')
    pose_stamped_4 = RigidTransformToPoseStamped(T_world_grasp)
    pose_stamped_4.pose.position.y = pose_stamped_4.pose.position.y + 0.2

    plan_4 = planner.plan_to_pose(pose_stamped_4)

    planner.execute_plan(plan_4)

    print('----------OPENING GRIPPER----------')
    open_gripper()

    print('---------- MOVING TO UP POSITION NBR 2 ----------')
    pose_stamped_5 = pose_stamped_4
    pose_stamped_5.pose.position.z = pose_stamped_5.pose.position.z + 0.3
    pose_stamped_5.pose.position.y = pose_stamped_5.pose.position.y - 0.2

    plan_5 = planner.plan_to_pose(pose_stamped_5)

    planner.execute_plan(plan_5)



def parse_args():
    """
    Pretty self explanatory tbh
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', type=str, default='gearbox', help=
        """Which Object you\'re trying to pick up.  Options: gearbox, nozzle, pawn.
        Default: gearbox"""
    )
    parser.add_argument('-n_vert', type=int, default=1000, help=
        'How many vertices you want to sample on the object surface.  Default: 1000'
    )
    parser.add_argument('-n_facets', type=int, default=32, help=
        """You will approximate the friction cone as a set of n_facets vectors along
        the surface.  This way, to check if a vector is within the friction cone, all
        you have to do is check if that vector can be represented by a POSITIVE
        linear combination of the n_facets vectors.  Default: 32"""
    )
    parser.add_argument('-n_grasps', type=int, default=500, help=
        'How many grasps you want to sample.  Default: 500')
    parser.add_argument('-n_execute', type=int, default=5, help=
        'How many grasps you want to execute.  Default: 5')
    parser.add_argument('-metric', '-m', type=str, default='compute_force_closure', help=
        """Which grasp metric in grasp_metrics.py to use.
        Options: compute_force_closure, compute_gravity_resistance, compute_custom_metric"""
    )
    parser.add_argument('-arm', '-a', type=str, default='right', help=
        'Options: left, right.  Default: right'
    )
    parser.add_argument('--baxter', action='store_true', help=
        """If you don\'t use this flag, you will only visualize the grasps.  This is
        so you can run this on your laptop"""
    )
    parser.add_argument('--debug', action='store_true', help=
        'Whether or not to use a random seed'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.debug:
        np.random.seed(0)

    print('starting main')
    rospy.init_node('main_node')


    # Mesh loading and pre-processing
    mesh = trimesh.load_mesh('objects/{}.obj'.format(args.obj))
    

    
    #mesh.apply_transform(T_world_obj.matrix)
    mesh.fix_normals()


    # This policy takes a mesh and returns the best actions to execute on the robot
    grasping_policy = GraspingPolicy(
        args.n_vert,
        args.n_grasps,
        args.n_execute,
        args.n_facets,
        args.metric
    )

    # Each grasp is represented by T_grasp_world, a RigidTransform defining the
    # position of the end effector

    # Execute each grasp on the baxter / sawyer
    if args.baxter:
        T_world_obj = lookup_transform(args.obj)
        print('T_world_obj')
        print(T_world_obj)
        print('')
        gripper = baxter_gripper.Gripper(args.arm)
        gripper.calibrate()
        planner = PathPlanner('{}_arm'.format(args.arm))

        T_obj_grasps = grasping_policy.top_n_actions(mesh, args.obj)
        T_world_grasps = []

        for i, Tog in enumerate(T_obj_grasps):
            # print('GRASP # {}'.format(i))

            ## computing the final positions ##
            T_wo = T_world_obj.matrix
            T_og = Tog.matrix
            T_wg = np.matmul(T_wo, T_og)

            T_world_grasps.append(RigidTransform(T_wg[:3, :3], T_wg[:3, 3], 'world', 'gripper'))

            print('T_world_grasp')
            print(T_world_grasps[i])
            print('')


        for T_world_grasp in T_world_grasps:
            repeat = True
            #print('GRASP')


            while repeat:
                execute_grasp(T_world_grasp, planner, gripper)
                repeat = raw_input("repeat? [y|n] ") == 'y'

    else:
        T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)
