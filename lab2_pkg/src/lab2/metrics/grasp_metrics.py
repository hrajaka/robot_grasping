#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for EE106B grasp planning lab
Author: Chris Correa
"""
# may need more imports
import numpy as np
from lab2.utils import vec, adj, look_at_general, hat
import cvxpy as cvx
import math
import scipy

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Compute the force closure of some object at contacts, with normal vectors
    stored in normals You can use the line method described in HW2.  if you do you
    will not need num_facets

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # to be in force closure with two contact points, we need the line between the two points to be in the friction cones
    # we compute the angle between the connecting line and the normal
    # if this angle is smaller than pi/2 - arctan(mu), then it's in force closure

    ## checking for first point of contact ##
    vec_between_vertices = vertices[1]-vertices[0]

    angle = np.arccos( np.dot(normals[0], vec_between_vertices) / (np.linalg.norm(normals[0]) * np.linalg.norm(vec_between_vertices)))

    if abs(angle) >= abs(np.arctan(mu)):
        return 0

    ## checking for second point of contact ##
    vec_between_vertices = vertices[0]-vertices[1]
    angle = np.arccos( np.dot(normals[1], vec_between_vertices) / (np.linalg.norm(normals[1]) * np.linalg.norm(vec_between_vertices)))

    if abs(angle) >= abs(np.arctan(mu)):
        return 0


    return 1

    # raise NotImplementedError

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """
    defined in the book on page 219.  Compute the grasp map given the contact
    points and their surface normals

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient

    Returns
    -------
    :obj:`numpy.ndarray` grasp map
    """

    B1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1]])

    B2 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 1]])

    g1 = np.zeros((4,4))
    g1[:3, :3] = hat(normals[0])
    g1[:3, 3] = vertices[0]
    g1[3, 3] = 1
    G1 = np.matmul(adj(g1.T), B1)

    g12= np.zeros((4,4))
    g2[:3, :3] = hat(normals[0])
    g2[:3, 3] = vertices[0]
    g2[3, 3] = 1
    G2 = np.matmul(adj(g1.T), B2)

    G = [G1, G2]

    return G
    # raise NotImplementedError

def contact_forces_exist(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute whether the given grasp (at contacts with surface normals) can produce
    the desired_wrench.  will be used for gravity resistance.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors
        will be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : :obj:`numpy.ndarray`
        potential wrench to be produced

    Returns
    -------
    bool : whether contact forces can produce the desired_wrench on the object
    """
    # we look for a solution to the system: desired_wrench = G @ f

    ## we compute the wrench ##
    # f = np.linalg.lstsq(G, desired_wrench)
    f = scipy.optimize.nnls(G, desired_wrench) # to get non negative solution

    # WARNING this might not work because the least square solution might not work, but other solutions might be in the friction cone
    # cf a comment by Ryan OGorman on Piazza

    ## we check if it belongs to the friction cone ##
    forces = [f[:4], f[4:]]
    for force in forces:
        # condition on the tangential forces
        if np.sqrt(force[0]**2 + force[1]**2) > mu * force[2]:
            return False

        # condition on the normal force
        if force[2] <= 0:
            return False

        # condition on the torque
        if abs(force[4]) > gamma * force[2]:
            return False

    return True

    # raise NotImplementedError

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    Gravity produces some wrench on your object.  Computes whether the grasp can
    produce and equal and opposite wrench

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will
        be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE (contact forces exist may be useful here)

    ## we compute the grasp matrix ##
    G = get_grasp_map(vertices, normals, num_facets, mu, gamma)

    ## we build the gravity wrench and see if it can be resisted ##
    gravity_wrench = np.array([0, 0, -9.81*object_mass, 0, 0, 0]) # is the torque non zero? don't think so because applied at center of mass but make sure

    # WARNING: cf comment around line 150 about why this might not work
    can_resist = contact_forces_exist(vertices, normals, num_facets, mu, gamma, gravity_wrench)

    return can_resist

    # raise NotImplementedError

def compute_custom_metric(vertices, normals, num_facets, mu, gamma, object_mass):
    """
    I suggest Ferrari Canny, but feel free to do anything other metric you find.

    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors will
        be along the friction cone boundary
    mu : float
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object

    Returns
    -------
    float : quality of the grasp
    """
    # YOUR CODE HERE :)

    return 42
    # raise NotImplementedError
