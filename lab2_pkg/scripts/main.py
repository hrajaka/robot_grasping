#!/home/cc/ee106b/sp19/class/ee106b-abj/python-virtual-environments/env/bin/python
# #!/home/cc/ee106b/sp19/class/ee106b-aai/virtualenvironment/my_new_app/bin/python

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
    rot = RigidTransform.rotation_from_quaternion(tag_rot)
    return RigidTransform(rot, tag_pos, to_frame=from_frame, from_frame=to_frame)

def execute_grasp(T_grasp_world, planner, gripper, T_obj_world):
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
    open_gripper()

    ## computing desired pose of gripper in world frame ##
    print(T_grasp_world) #this one is probably wrong
    print(T_obj_world) #this one is good (verified manually)
    g_T_grasp_world = np.array([[T_grasp_world.rotation[0,0], T_grasp_world.rotation[0,1], T_grasp_world.rotation[0,2], T_grasp_world.translation[0]], 
                                [T_grasp_world.rotation[1,0], T_grasp_world.rotation[1,1], T_grasp_world.rotation[1,2], T_grasp_world.translation[1]],
                                [T_grasp_world.rotation[2,0], T_grasp_world.rotation[2,1], T_grasp_world.rotation[2,2], T_grasp_world.translation[2]],
                                [0, 0, 0, 1]])

    g_T_obj_world = np.array([[T_obj_world.rotation[0,0], T_obj_world.rotation[0,1], T_obj_world.rotation[0,2], T_obj_world.translation[0]], 
                              [T_obj_world.rotation[1,0], T_obj_world.rotation[1,1], T_obj_world.rotation[1,2], T_obj_world.translation[1]],
                              [T_obj_world.rotation[2,0], T_obj_world.rotation[2,1], T_obj_world.rotation[2,2], T_obj_world.translation[2]],
                              [0, 0, 0, 1]])

    rigid_transfo_gripper_in_base = np.matmul(g_T_grasp_world, g_T_obj_world)
    p = rigid_transfo_gripper_in_base[0:-1,3]
    print('desired position of gripper in world frame: ', p)

    ## generating the PoseStamped (should stop a bit before the part) ##
    pose_stamped = PoseStamped()
    # todo

    ## creating the plan ##


    ## executing the plan ##


    ## generating the PoseStamped (the one that goes to the part) ##
    pose_stamped = PoseStamped()
    # todo

    ## creating the plan ##


    ## executing the plan ##


    ## closing gripper ##
    close_gripper()

    ## generating the PoseStamped (the one that lifts the part) ##
    pose_stamped = PoseStamped()
    # todo

    ## creating the plan ##


    ## executing the plan ##


    raise NotImplementedError

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
    #T_obj_world = lookup_transform(args.obj)
    #print(T_obj_world)
    # mesh.apply_transform(T_obj_world.matrix)
    mesh.fix_normals()


    # This policy takes a mesh and returns the best actions to execute on the robot
    grasping_policy = GraspingPolicy(
        args.n_vert, 
        args.n_grasps, 
        args.n_execute, 
        args.n_facets, 
        args.metric
    )

    G_rot = np.eye(3)
    G_trans = np.array([0.02, 0, 0])
    G = RigidTransform(G_rot, G_trans, 'object', 'gripper')

    grasping_policy.vis_transform(mesh, G)

    # Each grasp is represented by T_grasp_world, a RigidTransform defining the 
    # position of the end effector

    # Execute each grasp on the baxter / sawyer
    if args.baxter:
        gripper = baxter_gripper.Gripper(args.arm)
        planner = PathPlanner('{}_arm'.format(args.arm))

        T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)
        # WARNING: they are probably wrong
        # What would be great is to visualize them in the GUI

        for T_grasp_world in T_grasp_worlds:
            repeat = True
            while repeat:
                execute_grasp(T_grasp_world, planner, gripper, T_obj_world)
                repeat = raw_input("repeat? [y|n] ") == 'y'
    else:
        T_grasp_worlds = grasping_policy.top_n_actions(mesh, args.obj)