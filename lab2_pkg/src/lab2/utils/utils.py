#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Utils for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np
from numpy import linalg

from math import sin, cos, atan2, sqrt
try:
    from geometry_msgs.msg._Point import Point
    import tf.transformations as tfs
    from geometry_msgs.msg import Pose, PoseStamped
    ros_enabled = True
except:
    ros_enabled = False

def g_inv(g):
    R = g[0:3,0:3]
    p = g[0:3,3]

    g_inverse = g
    g_inverse[0:3,0:3] = R.T
    g_inverse[0:3,3] = -1*np.matmul(R.T, p)
    return g_inverse

def length(vec):
    """
    Returns the length of a 1 dimensional numpy vector

    Parameters
    ----------
    vec : nx1 :obj:`numpy.ndarray`

    Returns
    -------
    float
        ||vec||_2^2
    """
    return sqrt(vec.dot(vec))

def normalize(vec, axis=None):
    """
    Returns a normalized version of a numpy vector

    Parameters
    ----------
    vec : nx :obj:`numpy.ndarray

    Returns
    -------
    nx :obj:`numpy.ndarray`
    """
    return vec / length(vec) if axis == None else vec / np.linalg.norm(vec, axis=axis).reshape(-1,1)

def joint_array_to_dict(vel_torque_array, limb):
    """
    the baxter interface requires you to send the joint velocities / torques
    as a dictionary, this turns and array of velocities and torques into a 
    dictionary with joint names.

    Parameters
    ----------
    vel_torque_array : 7x1 :obj:`numpy.ndarray`
        numpy array of velocities or torques to be sent to the baxter
    limb : :obj:`baxter_interface.Limb`
        Limb object

    Returns
    -------
    :obj:`dict` of string->float
        mapping of joint names to joint velocities / torques
    """
    vel_torque_dict = dict()
    for i, joint in enumerate(limb.joint_names()):
        vel_torque_dict[joint] = vel_torque_array[i,0]
    return vel_torque_dict

def vec(*args):
    """
    all purpose function to get a numpy array of random things.  you can pass
    in a list, tuple, ROS Point message.  you can also pass in:
    vec(1,2,3,4,5,6) which will return a numpy array of each of the elements 
    passed in: np.array([1,2,3,4,5,6])
    """
    if len(args) == 1:
        if type(args[0]) == tuple:
            return np.array(args[0])
        elif ros_enabled and type(args[0]) == Point:
            return np.array((args[0].x, args[0].y, args[0].z))
        else:
            return np.array(args)
    else:
        return np.array(args)

def hat(v):
    """
    See https://en.wikipedia.org/wiki/Hat_operator or the MLS book

    Parameters
    ----------
    v : :obj:`numpy.ndarrray`
        vector form of shape 3x1, 3x, 6x1, or 6x

    Returns
    -------
    3x3 or 6x6 :obj:`numpy.ndarray`
        hat version of the vector v
    """
    if v.shape == (3, 1) or v.shape == (3,):
        return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
    elif v.shape == (6, 1) or v.shape == (6,):
        return np.array([
                [0, -v[5], v[4], v[0]],
                [v[5], 0, -v[3], v[1]],
                [-v[4], v[3], 0, v[2]],
                [0, 0, 0, 0]
            ])
    else:
        raise ValueError

def adj(g):
    """
    Adjoint of a rotation matrix.  See the MLS book

    Parameters
    ----------
    g : 4x4 :obj:`numpy.ndarray`
        Rotation matrix

    Returns
    -------
    6x6 :obj:`numpy.ndarray` 
    """
    if g.shape != (4, 4):
        raise ValueError

    R = g[0:3,0:3]
    p = g[0:3,3]
    result = np.zeros((6, 6))
    result[0:3,0:3] = R
    result[0:3,3:6] = np.matmul(hat(p), R)
    result[3:6,3:6] = R
    return result

def twist_from_tf(g):
    """
    Returns the twist version of a 2D rotation matrix
    Parameters
    ----------
    g : 3x3 :obj:`numpy.ndarray`
        2D rotation matrix

    Returns
    -------
    3x :obj:`numpy.ndarray`
    """
    return vec(g[0,2], g[1,2], atan2(g[1,0], g[0,0]))

def rotation2d(theta):
    """
    2D rotation matrix from a single theta around the origin

    Parameters
    ----------
    theta : float

    Returns
    -------
    2x2 :obj:`numpy.ndarray`
    """
    return np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ])

def rigid(twist):
    """
    Returns a 3x3 Rotation Matrix version of a 2D twist

    Parameters
    ----------
    twist : 3x1 :obj:`numpy.ndarray`

    Returns
    -------
    3x3 :obj:`numpy.ndarray`
    """
    return np.array([
            [cos(twist[2]), -sin(twist[2]), twist[0]],
            [sin(twist[2]), cos(twist[2]), twist[1]],
            [0, 0, 1]
        ])

def look_at_general(origin, direction):
    """
    Creates a 3D Rotation Matrix at the origin such that the z axis is the same
    as the direction specified.  There are infinitely many of such matrices, 
    but we choose the one where the x axis is as vertical as possible.  

    Parameters
    ----------
    origin : 3x1 :obj:`numpy.ndarray`
    x : 3x1 :obj:`numpy.ndarray`

    Returns
    -------
    4x4 :obj:`numpy.ndarray`
    """
    up = vec(0,0,1)
    z = normalize(direction)
    x = normalize(np.cross(up, z))
    y = np.cross(z, x) 

    result = np.eye(4)
    result[0:3,0] = x
    result[0:3,1] = y
    result[0:3,2] = z
    result[0:3,3] = origin
    return result

def create_pose_from_rigid_transform(g):
    """
    takes a rotation matrix and turns it into a ROS Pose

    Parameters
    ----------
    g : 4x4 : :obj:`numpy.ndarray`

    Returns
    -------
    :obj:`geometry_msgs.msg.Pose`
    """
    position = tfs.translation_from_matrix(g)
    quaternion = tfs.quaternion_from_matrix(g)
    wpose = Pose()
    wpose.position.x = position[0]
    wpose.position.y = position[1]
    wpose.position.z = position[2]
    wpose.orientation.x = quaternion[0]
    wpose.orientation.y = quaternion[1]
    wpose.orientation.z = quaternion[2]
    wpose.orientation.w = quaternion[3]
    return wpose


####################  Stuff from lab3 of ee206a  #####################

def skew_3d(omega):
    """
    Converts a rotation vector in 3D to its corresponding skew-symmetric matrix.
    
    Args:
    omega - (3,) ndarray: the rotation vector
    
    Returns:
    omega_hat - (3,3) ndarray: the corresponding skew symmetric matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')
    
    #YOUR CODE HERE

    omega_hat = np.array([ [ 0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0] ])

    return omega_hat

def rotation_2d(theta):
    """
    Computes a 2D rotation matrix given the angle of rotation.
    
    Args:
    theta: the angle of rotation
    
    Returns:
    rot - (2,2) ndarray: the resulting rotation matrix
    """
    
    #YOUR CODE HERE

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ])

    return rot

def rotation_3d(omega, theta):
    """
    Computes a 3D rotation matrix given a rotation axis and angle of rotation.
    
    Args:
    omega - (3,) ndarray: the axis of rotation
    theta: the angle of rotation
    
    Returns:
    rot - (3,3) ndarray: the resulting rotation matrix
    """
    if not omega.shape == (3,):
        raise TypeError('omega must be a 3-vector')
    
    #YOUR CODE HERE
    #if (np.linalg.norm(omega) != 1):
    #    omega = omega / np.linalg.norm(omega)

    rot = np.eye(3) + (skew_3d(omega)*np.sin(np.linalg.norm(omega)*theta))/np.linalg.norm(omega) + (np.matmul(skew_3d(omega),skew_3d(omega)))*(1 - np.cos(np.linalg.norm(omega)*theta))/(np.linalg.norm(omega)**2)

    return rot

def hat_2d(xi):
    """
    Converts a 2D twist to its corresponding 3x3 matrix representation
    
    Args:
    xi - (3,) ndarray: the 2D twist
    
    Returns:
    xi_hat - (3,3) ndarray: the resulting 3x3 matrix
    """
    if not xi.shape == (3,):
        raise TypeError('omega must be a 3-vector')

    #YOUR CODE HERE

    xi_hat = np.array([[0, -xi[2], xi[0]], [xi[2], 0, xi[1]], [0, 0, 0]])

    return xi_hat

def hat_3d(xi):
    """
    Converts a 3D twist to its corresponding 4x4 matrix representation
    
    Args:
    xi - (6,) ndarray: the 3D twist
    
    Returns:
    xi_hat - (4,4) ndarray: the corresponding 4x4 matrix
    """
    if not xi.shape == (6,):
        raise TypeError('xi must be a 6-vector')

    #YOUR CODE HERE

    xi_hat = np.array([ [0, -xi[5], xi[4], xi[0]], [xi[5], 0, -xi[3], xi[1]], [-xi[4], xi[3], 0, xi[2]], [0, 0, 0 ,0] ])

    return xi_hat

def homog_2d(xi, theta):
    """
    Computes a 3x3 homogeneous transformation matrix given a 2D twist and a 
    joint displacement
    
    Args:
    xi - (3,) ndarray: the 2D twist
    theta: the joint displacement
    
    Returns:
    g - (3,3) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (3,):
        raise TypeError('xi must be a 3-vector')

    #YOUR CODE HERE
    w = xi[2]

    R = np.array([[np.cos(w*theta), -np.sin(w*theta)], [np.sin(w*theta), np.cos(w*theta)] ])
    pA = np.array([ [1 - np.cos(w*theta), np.sin(w*theta)], [-np.sin(w*theta), 1 - np.cos(w*theta)] ])
    pB = np.array([[0, -1], [1, 0]])
    pC = np.array([ [xi[0]/w], [xi[1]/w] ])

    p = np.matmul(pA, pB)
    p = np.matmul(p, pC)

    g = np.hstack((R, p))
    g = np.vstack((g, [0,0,1]))

    return g

def homog_3d(xi, theta):
    """
    Computes a 4x4 homogeneous transformation matrix given a 3D twist and a 
    joint displacement.
    
    Args:
    xi - (6,) ndarray: the 3D twist
    theta: the joint displacement

    Returns:
    g - (4,4) ndarary: the resulting homogeneous transformation matrix
    """
    if not xi.shape == (6,):
        raise TypeError('xi must be a 6-vector')

    #YOUR CODE HERE


    v = xi[:3]
    omega = xi[3:]

    omega_hat = skew_3d(omega)

    exp_omega_theta = rotation_3d(omega, theta)

    other_term = (np.eye(3) - exp_omega_theta)
    toto =  np.matmul(omega_hat, [[v[0]] , [v[1]] , [v[2]]])
    other_term = np.matmul(other_term, toto)
   
    wwT = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            wwT[i,j] = omega[i]*omega[j]

    other_term2 = wwT
    other_term2 = np.matmul(other_term2, [[v[0]] , [v[1]] , [v[2]]])
    other_term2 = other_term2 *theta

    other_term = other_term + other_term2
    other_term = other_term / np.linalg.norm(omega)**2


    g = np.hstack((exp_omega_theta, other_term))
    g = np.vstack((g, [0,0,0,1]))
        

    return g

def prod_exp(xi, theta):
    """
    Computes the product of exponentials for a kinematic chain, given 
    the twists and displacements for each joint.
    
    Args:
    xi - (6,N) ndarray: the twists for each joint
    theta - (N,) ndarray: the displacement of each joint
    
    Returns:
    g - (4,4) ndarray: the resulting homogeneous transformation matrix
    """
    if not xi.shape[0] == 6:
        raise TypeError('xi must be a 6xN')

    g = np.eye(4)

    #YOUR CODE HERE
    for i in range(xi.shape[1]):
        g = np.matmul(g,homog_3d(xi[:,i], theta[i]))

    return g