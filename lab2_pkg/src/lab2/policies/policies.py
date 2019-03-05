#!/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasping Policy for EE106B grasp planning lab
Author: Chris Correa
"""
import numpy as np

# Autolab imports
from autolab_core import RigidTransform
import trimesh
from visualization import Visualizer3D as vis3d
import random
import matplotlib as plt

# 106B lab imports
from lab2.metrics import (
    compute_force_closure,
    compute_gravity_resistance,
    compute_custom_metric
)
from lab2.utils import length, normalize

# YOUR CODE HERE
# probably don't need to change these (BUT confirm that they're correct)
MAX_HAND_DISTANCE = .04
MIN_HAND_DISTANCE = .01
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1

# TODO
OBJECT_MASS = {'gearbox': .25, 'nozzle': .25, 'pawn': .25}


class GraspingPolicy():
    def __init__(self, n_vert, n_grasps, n_execute, n_facets, metric_name):
        """
        Parameters
        ----------
        n_vert : int
            We are sampling vertices on the surface of the object, and will use pairs of
            these vertices as grasp candidates
        n_grasps : int
            how many grasps to sample.  Each grasp is a pair of vertices
        n_execute : int
            how many grasps to return in policy.action()
        n_facets : int
            how many facets should be used to approximate the friction cone between the
            finger and the object
        metric_name : string
            name of one of the function in src/lab2/metrics/metrics.py
        """
        self.n_vert = n_vert
        self.n_grasps = n_grasps
        self.n_facets = n_facets
        self.n_execute = n_execute
        # This is a function, one of the functions in src/lab2/metrics/metrics.py
        self.metric = eval(metric_name)
        self.metric_name = metric_name

    def vertices_to_baxter_hand_pose(grasp_vertices, approach_direction):
        """
        takes the contacts positions in the object frame and returns the hand pose T_obj_gripper
        BE CAREFUL ABOUT THE FROM FRAME AND TO FRAME.  the RigidTransform class' frames are
        weird.

        Parameters
        ----------
        grasp_vertices : 2x3 :obj:`numpy.ndarray`
            position of the fingers in object frame
        approach_direction : 3x' :obj:`numpy.ndarray`
            there are multiple grasps that go through contact1 and contact2.  This describes which
            orientation the hand should be in

        Returns
        -------
        :obj:`autolab_core:RigidTransform` Hand pose in the object frame
        """
        # parameters required to create a autolab_core:RigidTransform:
        # - rotation (aka 3x3 rotation matrix)
        # - translation (aka 3x1 vector)
        # - from_frame (aka str)
        # - to_frame (aka str)

        # the position we want to go to is in the middle of the two points
        # the orientation we want to have is the approach_direction??

        # apparently it is supposed to be in the object coordinates (cf function execute_grasp in main.py)
        raise NotImplementedError

    def sample_grasps(self, vertices, normals):
        """
        Samples a bunch of candidate grasps.  You should randomly choose pairs of vertices and throw out
        pairs which are too big for the gripper, or too close too the table.  You should throw out vertices
        which are lower than ~3cm of the table.  You may want to change this.  Returns the pairs of
        grasp vertices and grasp normals (the normals at the grasp vertices)

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
            mesh vertices
        normals : nx3 :obj:`numpy.ndarray`
            mesh normals
        T_ar_object : :obj:`autolab_core.RigidTransform`
            transform from the AR tag on the paper to the object

        Returns
        -------
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps vertices.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector and there are n_grasps of them, hence the shape n_graspsx2x3
        n_graspsx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        """
        # find the distance the vertices
        # if bigger than gripper, remove it from list
        # find the distance to the table
        # if too small remove from list

        grasp_vertices = []
        grasp_normals = []
        nbr_grasps_found = 0

        for i in range(self.n_grasps):
            hasFoundValidGrasp = False
            while not hasFoundValidGrasp:
                # pick two random points
                idx1 = random.randint(0, len(vertices)-1)
                idx2 = random.randint(0, len(vertices)-1)
                
                if idx1 == idx2:
                    continue

                # now we compute the distance
                distance = np.linalg.norm(vertices[idx1] - vertices[idx2])
                if distance > MAX_HAND_DISTANCE or distance < MIN_HAND_DISTANCE:
                    continue
                # checking if too close to ground
                if vertices[idx1][2] < 0.03 or vertices[idx2][2] < 0.03:
                    continue

                # at this point it means we have a valid pair of points
                hasFoundValidGrasp = True
            curr_grasp_vertices = [vertices[idx1], vertices[idx2]]
            curr_grasp_normals = [normals[idx1], normals[idx2]]
            grasp_vertices.append(curr_grasp_vertices)
            grasp_normals.append(curr_grasp_normals)

        grasp_vertices = np.array(grasp_vertices)
        grasp_normals = np.array(grasp_normals)

        return grasp_vertices, grasp_normals


    def score_grasps(self, grasp_vertices, grasp_normals, object_mass):
        """
        takes mesh and returns pairs of contacts and the quality of grasp between the contacts, sorted by quality

        Parameters
        ----------
        grasp_vertices : n_graspsx2x3 :obj:`numpy.ndarray`
            grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3
        grasp_normals : mx2x3 :obj:`numpy.ndarray`
            grasps normals.  Each grasp containts two contact points.  Each vertex normal
            is a 3 dimensional vector, and there are n_grasps of them, hence the shape n_graspsx2x3

        Returns
        -------
        :obj:`list` of int
            grasp quality for each
        """
        grasp_qualities = []

        if self.metric_name == 'compute_force_closure':
            for i in range(grasp_vertices.shape[0]):
                grasp_qualities.append(compute_force_closure(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass))
        elif self.metric_name == 'compute_gravity_resistance':
            for i in range(grasp_vertices.shape[0]):
                grasp_qualities.append(compute_gravity_resistance(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass))
        else:
            for i in range(grasp_vertices.shape[0]):
                grasp_qualities.append(compute_custom_metric(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass))
        
        return grasp_qualities

    def vis(self, mesh, grasp_vertices, grasp_qualities):
        """
        Pass in any grasp and its associated grasp quality.  this function will plot
        each grasp on the object and plot the grasps as a bar between the points, with
        colored dots on the line endpoints representing the grasp quality associated
        with each grasp

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        grasp_vertices : mx2x3 :obj:`numpy.ndarray`
            m grasps.  Each grasp containts two contact points.  Each contact point
            is a 3 dimensional vector, hence the shape mx2x3
        grasp_qualities : mx' :obj:`numpy.ndarray`
            vector of grasp qualities for each grasp
        """
        vis3d.mesh(mesh)

        dirs = normalize(grasp_vertices[:,0] - grasp_vertices[:,1], axis=1)


        midpoints = (grasp_vertices[:,0] + grasp_vertices[:,1]) / 2
        grasp_endpoints = np.zeros(grasp_vertices.shape)
        grasp_vertices[:,0] = midpoints + dirs*MAX_HAND_DISTANCE/2
        grasp_vertices[:,1] = midpoints - dirs*MAX_HAND_DISTANCE/2

        for grasp, quality in zip(grasp_vertices, grasp_qualities):
            color = [min(1, 2*(1-quality)), min(1, 2*quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)
        vis3d.show()

    def top_n_actions(self, mesh, obj_name, vis=True):
        """
        Takes in a mesh, samples a bunch of grasps on the mesh, evaluates them using the
        metric given in the constructor, and returns the best grasps for the mesh.  SHOULD
        RETURN GRASPS IN ORDER OF THEIR GRASP QUALITY.

        You should try to use mesh.mass to get the mass of the object.  You should check the
        output of this, because from this
        https://github.com/BerkeleyAutomation/trimesh/blob/master/trimesh/base.py#L2203
        it would appear that the mass is approximated using the volume of the object.  If it
        is not returning reasonable results, you can manually weight the objects and store
        them in the dictionary at the top of the file.

        Parameters
        ----------
        mesh : :obj:`Trimesh`
        vis : bool
            Whether or not to visualize the top grasps

        Returns
        -------
        :obj:`list` of :obj:`autolab_core.RigidTransform`
            the matrices T_grasp_world, which represents the hand poses of the baxter / sawyer
            which would result in the fingers being placed at the vertices of the best grasps
        """
        # Some objects have vertices in odd places, so you should sample evenly across
        # the mesh to get nicer candidate grasp points using trimesh.sample.sample_surface_even()

        ## computing vertices ##
        # vertices, ids = trimesh.sample.sample_surface(mesh, self.n_vert)
        vertices, ids = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        normals = mesh.face_normals[ids]

        # do stuff with convex hull

        ## sampling some grasps ##
        grasp_vertices, grasp_normals = self.sample_grasps(vertices, normals)
        object_mass = OBJECT_MASS[obj_name]

        ## computing grasp qualities and finding the n best ##
        grasp_qualities = self.score_grasps(grasp_vertices, grasp_normals, object_mass)
        
        ## visualizing all grasps ##
        self.vis(mesh, grasp_vertices, np.array(grasp_qualities))
        
        ## keeping only the best n_execute ##
        grasp_vertices = list(grasp_vertices)
        grasp_normals = list(grasp_normals)
        best_grasp_vertices = []
        best_grasp_normals = []
        best_grasp_qualities = []
        nbr_best_found = 0
        
        while nbr_best_found < self.n_execute:
            idx_max = np.argmax(grasp_qualities)
            best_grasp_vertices.append(grasp_vertices[idx_max])
            best_grasp_normals.append(grasp_normals[idx_max])
            best_grasp_qualities.append(grasp_qualities[idx_max])
            grasp_vertices.pop(idx_max)
            grasp_normals.pop(idx_max)
            grasp_qualities.pop(idx_max)
            nbr_best_found += 1 

        best_grasp_vertices = best_grasp_vertices[::-1][:self.n_execute]
        best_grasp_normals = best_grasp_normals[::-1][:self.n_execute]
        best_grasp_qualities = best_grasp_qualities[::-1][:self.n_execute]

        self.vis(mesh, np.array(best_grasp_vertices), np.array(best_grasp_qualities))



        ## generating the hand poses ##
        # approach_direction = ?? Maybe something orthogonal to the line between the two points
        # and in what frame is it??
        # they talk about it on Piazza but not clear
        hand_poses = self.vertices_to_baxter_hand_pose(grasp_vertices, approach_direction)

        return hand_poses
