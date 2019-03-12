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
import matplotlib.pyplot as plt

# 106B lab imports
from lab2.metrics import (
    compute_force_closure,
    compute_gravity_resistance,
    compute_custom_metric,
    robust_force_closure
)
from lab2.utils import length, normalize, rotation_3d


MAX_HAND_DISTANCE = 0.085
MIN_HAND_DISTANCE = 0.045
CONTACT_MU = 0.5
CONTACT_GAMMA = 0.1
finger_length = 0.1


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

    def vertices_to_baxter_hand_pose(self, grasp_vertices, approach_direction, obj_name):
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
        
        midpoint = (grasp_vertices[0] + grasp_vertices[1]) / 2

        gripper_half_width = MAX_HAND_DISTANCE / 2
        
        z = normalize(approach_direction)
        y = normalize(grasp_vertices[0] - grasp_vertices[1])
        x = np.cross(y, z)

        rot_mat_opposite = np.array([x, y, z]).T
        p_opposite = midpoint

        rot_mat = rot_mat_opposite.T
        p = - np.matmul(rot_mat_opposite.T, p_opposite)

        rigid_trans = RigidTransform(rot_mat_opposite, p_opposite, to_frame='right_gripper', from_frame=obj_name) 

        return rigid_trans


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
        grasp_vertices = []
        grasp_normals = []
        nbr_grasps_found = 0
        z_table = min(vertices[:, 2]) #not exactly true but close
        #print('z_table', z_table)

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
                # if vertices[idx1][2] < 0.0 or vertices[idx2][2] < 0.0: #has to be changed when we apply the transform to the mesh
                if vertices[idx1][2] < z_table + 0.03 or vertices[idx2][2] < z_table + 0.03:
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
                grasp_qualities.append(compute_force_closure(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE))
        elif self.metric_name == 'compute_gravity_resistance':
            for i in range(grasp_vertices.shape[0]):
                grasp_qualities.append(compute_gravity_resistance(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass))
        else:
            for i in range(grasp_vertices.shape[0]):
                if i % 10 == 0:
                    print('testing vertex {} for robust force closure'.format(i))
                grasp_qualities.append(compute_custom_metric(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE))

        return grasp_qualities

    def vis(self, mesh, grasp_vertices, grasp_qualities, grasp_normals):
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

        middle_of_part = np.mean(np.mean(grasp_vertices, axis=1), axis=0)
        print(middle_of_part)
        vis3d.points(middle_of_part, scale=0.003)



        dirs = normalize(grasp_vertices[:,0] - grasp_vertices[:,1], axis=1)

        midpoints = (grasp_vertices[:,0] + grasp_vertices[:,1]) / 2
        grasp_endpoints = np.zeros(grasp_vertices.shape)
        grasp_endpoints[:,0] = midpoints + dirs*MAX_HAND_DISTANCE/2
        grasp_endpoints[:,1] = midpoints - dirs*MAX_HAND_DISTANCE/2

        n0 = np.zeros(grasp_endpoints.shape)
        n1 = np.zeros(grasp_endpoints.shape)

        normal_scale = 0.01
        n0[:, 0] = grasp_vertices[:, 0]
        n0[:, 1] = grasp_vertices[:, 0] + normal_scale * grasp_normals[:, 0]
        n1[:, 0] = grasp_vertices[:, 1]
        n1[:, 1] = grasp_vertices[:, 1] + normal_scale * grasp_normals[:, 1]

        for grasp, quality, normal0, normal1 in zip(grasp_endpoints, grasp_qualities, n0, n1):
            color = [min(1, 2*(1-quality)), min(1, 2*quality), 0, 1]
            vis3d.plot3d(grasp, color=color, tube_radius=.001)
            vis3d.plot3d(normal0, color=(0, 0, 0), tube_radius=.002)
            vis3d.plot3d(normal1, color=(0, 0, 0), tube_radius=.002)
        vis3d.show()

    def vis_transform(self, mesh, G_transform, vertices):
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
        L = MAX_HAND_DISTANCE / 2 # gripper half width

        # transform from gripper to contact 1
        G_gc1 = np.array([[1,  0,  0,    0],
                         [0,  0,  1, -1*L],
                         [0, -1,  0,    0],
                         [0,  0,  0,    1]])

        # transform from gripper to contact 2
        G_gc2 = np.array([[1,  0,  0,    0],
                         [0,  0, -1,    L],
                         [0,  1,  0,    0],
                         [0,  0,  0,    1]])

        G = G_transform.matrix

        print('G')
        print(G)

        G_oc1 = np.matmul(G, G_gc1)
        G_oc2 = np.matmul(G, G_gc2)


        scale = 0.01
        o = np.array([0, 0, 0, 1])
        x = np.array([scale, 0, 0, 1])
        y = np.array([0, scale, 0, 1])
        z = np.array([0, 0, scale, 1])

        ot = np.matmul(G, o)
        xt = np.matmul(G, x)
        yt = np.matmul(G, y)
        zt = np.matmul(G, z)


        o1 = np.matmul(G_oc1, o)
        x1 = np.matmul(G_oc1, x)
        y1 = np.matmul(G_oc1, y)
        z1 = np.matmul(G_oc1, z)

        o2 = np.matmul(G_oc2, o)
        x2 = np.matmul(G_oc2, x)
        y2 = np.matmul(G_oc2, y)
        z2 = np.matmul(G_oc2, z)

        vis3d.mesh(mesh, style='wireframe')


        
        #Plot origin axes
        x_axis = np.array([o, x])[:, :3]
        y_axis = np.array([o, y])[:, :3]
        z_axis = np.array([o, z])[:, :3]

        x_axis_t = np.array([ot, xt])[:, :3]
        y_axis_t = np.array([ot, yt])[:, :3]
        z_axis_t = np.array([ot, zt])[:, :3]

        x_axis_1 = np.array([o1, x1])[:, :3]
        y_axis_1 = np.array([o1, y1])[:, :3]
        z_axis_1 = np.array([o1, z1])[:, :3]

        x_axis_2 = np.array([o2, x2])[:, :3]
        y_axis_2 = np.array([o2, y2])[:, :3]
        z_axis_2 = np.array([o2, z2])[:, :3]


        vis3d.plot3d(x_axis, color=(0.5,0,0), tube_radius=0.001)
        vis3d.plot3d(y_axis, color=(0,0.5,0), tube_radius=0.001)
        vis3d.plot3d(z_axis, color=(0,0,0.5), tube_radius=0.001)

        vis3d.plot3d(x_axis_t, color=(255,0,0), tube_radius=0.001)
        vis3d.plot3d(y_axis_t, color=(0,255,0), tube_radius=0.001)
        vis3d.plot3d(z_axis_t, color=(0,0,255), tube_radius=0.001)

        vis3d.plot3d(x_axis_1, color=(255,0,0), tube_radius=0.001)
        vis3d.plot3d(y_axis_1, color=(0,255,0), tube_radius=0.001)
        vis3d.plot3d(z_axis_1, color=(0,0,255), tube_radius=0.001)

        vis3d.plot3d(x_axis_2, color=(255,0,0), tube_radius=0.001)
        vis3d.plot3d(y_axis_2, color=(0,255,0), tube_radius=0.001)
        vis3d.plot3d(z_axis_2, color=(0,0,255), tube_radius=0.001)

        vis3d.points(vertices[0], scale=0.003)
        vis3d.points(vertices[1], scale=0.003)

        vis3d.show()

    def compute_approach_direction(self, mesh, grasp_vertices, grasp_quality, grasp_normals):

        ## initalizing stuff ##
        visualize = True
        nb_directions_to_test = 6
        normal_scale = 0.01
        plane_normal = normalize(grasp_vertices[0] - grasp_vertices[1])
    
        midpoint = (grasp_vertices[0] + grasp_vertices[1]) / 2

        ## generating a certain number of approach directions ##
        theta = np.pi / nb_directions_to_test
        rot_mat = rotation_3d(-plane_normal, theta)

        horizontal_direction = normalize(np.cross(plane_normal, np.array([0, 0, 1])))
        directions_to_test = [horizontal_direction] #these are vectors
        approach_directions = [np.array([midpoint, midpoint + horizontal_direction * normal_scale])] #these are two points for visualization

        for i in range(nb_directions_to_test-1):
            directions_to_test.append(normalize(np.matmul(rot_mat, directions_to_test[-1])))
            approach_directions.append(np.array([midpoint, midpoint + directions_to_test[-1] * normal_scale]) )

        ## computing the palm position for each approach direction ##
        palm_positions = []
        for i in range(nb_directions_to_test):
            palm_positions.append(midpoint + finger_length * directions_to_test[i])


        if visualize:
            ## plotting the whole mesh ##
            vis3d.mesh(mesh, style='wireframe')

            ## computing and plotting midpoint and gripper position ##
            dirs = (grasp_vertices[0] - grasp_vertices[1]) / np.linalg.norm(grasp_vertices[0] - grasp_vertices[1])
            grasp_endpoints = np.zeros(grasp_vertices.shape)
            grasp_endpoints[0] = midpoint + dirs*MAX_HAND_DISTANCE/2
            grasp_endpoints[1] = midpoint - dirs*MAX_HAND_DISTANCE/2

            color = [min(1, 2*(1-grasp_quality)), min(1, 2*grasp_quality), 0, 1]
            vis3d.plot3d(grasp_endpoints, color=color, tube_radius=.001)
            vis3d.points(midpoint, scale=0.003)

            ## computing and plotting normals at contact points ##
            n0 = np.zeros(grasp_endpoints.shape)
            n1 = np.zeros(grasp_endpoints.shape)
            n0[0] = grasp_vertices[0]
            n0[1] = grasp_vertices[0] + normal_scale * grasp_normals[0]
            n1[0] = grasp_vertices[1]
            n1[1] = grasp_vertices[1] + normal_scale * grasp_normals[1]
            vis3d.plot3d(n0, color=(0, 0, 0), tube_radius=.002)
            vis3d.plot3d(n1, color=(0, 0, 0), tube_radius=.002)

            ## plotting normals the palm positions for each potential approach direction ##
            for i in range(nb_directions_to_test):
                vis3d.points(palm_positions[i], scale=0.003)

            vis3d.show()

        directions_to_test = [directions_to_test[3], directions_to_test[2], directions_to_test[4], directions_to_test[1], directions_to_test[5], directions_to_test[0]]
        palm_positions = [palm_positions[3], palm_positions[2], palm_positions[4], palm_positions[1], palm_positions[5], palm_positions[0]]

        ## checking if some approach direction is valid ##
        for i in range(nb_directions_to_test):
            if len(trimesh.intersections.mesh_plane(mesh, directions_to_test[i], palm_positions[i])) == 0:
                # it means the palm won't bump with part
                return directions_to_test[i]
        
        # it means all approach directions will bump with part 
        return -1


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

        ## old stuff ##
        # vertices, ids = trimesh.sample.sample_surface(mesh, self.n_vert)
        # vertices, ids = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        # convex_hull = trimesh.convex.convex_hull(mesh)
        # intersection = trimesh.boolean.intersection([mesh, convex_hull], engine='scad')

        ## computing vertices ##
        print('SAMPLING VERTICES')
        vertices, ids = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        normals = mesh.face_normals[ids] #face or vertex ????
        normals = -1 * normals

        ## sampling some grasps ##
        print('SAMPLING GRASPS')

        grasp_vertices, grasp_normals = self.sample_grasps(vertices, normals)
        object_mass = OBJECT_MASS[obj_name]


        ## computing grasp qualities and finding the n best ##
        print('COMPUTING GRASP QUALITIES')
        grasp_qualities = self.score_grasps(grasp_vertices, grasp_normals, object_mass)


        ## visualizing all grasps ##
        #self.vis(mesh, grasp_vertices, np.array(grasp_qualities), np.array(grasp_normals))

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

        best_grasp_vertices = np.array(best_grasp_vertices)
        best_grasp_qualities = np.array(best_grasp_qualities)
        best_grasp_normals = np.array(best_grasp_normals)

        print('BEST GRASPS:')
        print(best_grasp_qualities)
        print(best_grasp_vertices[0])
        print(best_grasp_vertices[1])
        print(best_grasp_vertices[2])
        print(best_grasp_vertices[3])
        print(best_grasp_vertices[4])
        ## visualizing the best grasps ##
        self.vis(mesh, best_grasp_vertices, best_grasp_qualities, best_grasp_normals)


        ## generating the hand poses ##
        print('GENERATING HAND POSES')
        hand_poses = []

        for i in range(self.n_execute):
            approach_dir = self.compute_approach_direction(mesh, best_grasp_vertices[i], best_grasp_qualities[i], best_grasp_normals[i])
            # WARNING: maybe we should take the opposite of approach_dir -> need to visualize it to make sure
            approach_dir = - approach_dir

            if type(approach_dir) == int:
                # it means the palm will bump in the part no matter from where it arrives
                print('Grasp not doable')
                raise Exception
            
            hand_poses.append(self.vertices_to_baxter_hand_pose(best_grasp_vertices[i], approach_dir, obj_name))
            self.vis_transform(mesh, hand_poses[-1], best_grasp_vertices[i])

        return hand_poses
    
    def parameter_sweep(self, mesh, obj_name):
        print('------------------------')
        print('RUNNING PARAMETER SWEEP')
        print('------------------------')

        print('SAMPLING VERTICES')
        vertices, ids = trimesh.sample.sample_surface_even(mesh, self.n_vert)
        normals = mesh.face_normals[ids] #face or vertex ????
        normals = -1 * normals

        ## sampling some grasps ##
        print('SAMPLING GRASPS FROM GEARBOX')

        grasp_vertices, grasp_normals = self.sample_grasps(vertices, normals)
        object_mass = OBJECT_MASS[obj_name]

        print('vertices:')
        print(grasp_vertices.shape)

        ## computing grasp qualities and finding the n best ##
        print('COMPUTING GRASP QUALITIES')
        grasp_qualities = []

        stds = np.array([0.005, 0.005, 0.1])
        for i in range(grasp_vertices.shape[0]):
            if i % 10 == 0:
                print('testing vertex {} for robust force closure'.format(i))
            grasp_qualities.append(robust_force_closure(grasp_vertices[i], grasp_normals[i], self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE, stds))
        grasp_qualities = np.array(grasp_qualities)
            

        
        v1 = grasp_vertices[:,0,:]
        v2 = grasp_vertices[:,1,:]
        q = np.array([grasp_qualities]).T
        
        data=np.hstack((v1,v2,q))
        print('v1x,v1y,v1z,v2x,v2y,v2z,quality')
        print('data:')
        print(data.shape)

        ind = np.argmax(grasp_qualities)
        print('index of max: {}'.format(ind))

        best_vertices = grasp_vertices[ind]
        best_normals = grasp_normals[ind]
        best_quality = grasp_qualities[ind]

        print('BEST RESULT:')
        print('vertices:')
        print(grasp_vertices[ind])
        print('normals')
        print(grasp_normals[ind])
        print('quality')
        print(grasp_qualities[ind])

        best_vertices = grasp_vertices[ind]
        best_normals = grasp_normals[ind]
        best_quality = grasp_qualities[ind]

        #EXPERIMENTAL RESULT FOR THIS GRASP
        SUCCESS_RATE = 0.8

        # sweep coefficient of friction
        num_test_points = 15
        min_len_test = np.linspace(stds[0]-0.005, stds[0]+0.005, num_test_points)
        max_len_test = np.linspace(stds[1]-0.005, stds[1]+0.005, num_test_points)
        mu_test =      np.linspace(stds[2]-0.1, stds[2]+0.1, num_test_points)
        
        new_qualities = []
        square_errors = []

        for mu in mu_test:
            new_stds = np.array([0.005, 0.005, mu])
            new_quality = robust_force_closure(best_vertices, best_normals, self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE, new_stds)
            new_qualities.append(new_quality)
            square_errors.append((SUCCESS_RATE - new_quality) ** 2)
        new_qualities = np.array(new_qualities)
        square_errors = np.array(square_errors)

        min_index = np.argmin(square_errors)
        print('best std of mu: {}'.format(mu_test[min_index]))

        plt.figure()
        plt.grid(True)
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.title('Varying standard deviation of coefficient of friction')
        plt.xlabel('std of mu')
        plt.ylabel('square-error')
        plt.plot(mu_test, square_errors, color='r')
        #plt.show()

        new_qualities = []
        square_errors = []

        for m in min_len_test:
            new_stds = np.array([m, 0.005, 0.1])
            new_quality = robust_force_closure(best_vertices, best_normals, self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE, new_stds)
            new_qualities.append(new_quality)
            square_errors.append((SUCCESS_RATE - new_quality) ** 2)
        new_qualities = np.array(new_qualities)
        square_errors = np.array(square_errors)

        min_index = np.argmin(square_errors)
        print('best std of min gripper length: {}'.format(min_len_test[min_index]))

        plt.figure()
        plt.grid(True)
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.title('Varying standard deviation of min gripper length')
        plt.xlabel('std of len')
        plt.ylabel('square-error')
        plt.plot(min_len_test, square_errors, color='r')

        new_qualities = []
        square_errors = []

        for m in max_len_test:
            new_stds = np.array([0.005, m, 0.1])
            new_quality = robust_force_closure(best_vertices, best_normals, self.n_facets, CONTACT_MU, CONTACT_GAMMA, object_mass, MIN_HAND_DISTANCE, MAX_HAND_DISTANCE, new_stds)
            new_qualities.append(new_quality)
            square_errors.append((SUCCESS_RATE - new_quality) ** 2)
        new_qualities = np.array(new_qualities)
        square_errors = np.array(square_errors)

        min_index = np.argmin(square_errors)
        print('best std of max gripper length: {}'.format(max_len_test[min_index]))

        plt.figure()
        plt.grid(True)
        plt.axvline(color='k')
        plt.axhline(color='k')
        plt.title('Varying standard deviation of max gripper length')
        plt.xlabel('std of len')
        plt.ylabel('square-error')
        plt.plot(min_len_test, square_errors, color='r')

        plt.show()
