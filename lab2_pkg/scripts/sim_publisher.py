#!/usr/bin/env python

#import roslib
#roslib.load_manifest('lab2_pkg')
import rospy
import math
import numpy as np
import tf
import geometry_msgs.msg
import sys

#from autolab_core import RigidTransform

'''
class ObjectTemplate(object):
    def __init__(self, name, ar_marker, R_ar_obj=np.eye(3), t_ar_obj=np.zeros(3)):
        """
        Struct for specifying object templates

        Parameters
        ----------
        name : string
            name of object
        ar_marker : string
            name of ar marker on object template
        R_ar_obj : 3x3 :obj:`numpy.ndarray`
            rotation between AR marker and object
        t_ar_obj : 3x' :obj:`numpy.ndarray`
            translation between AR marker and objectSS
        """
        self.name = name
        self.ar_marker = ar_marker
        self.T_ar_obj = RigidTransform(rotation=R_ar_obj, translation=t_ar_obj,
                                       from_frame=name, to_frame=ar_marker)

    @property
    def q_ar_obj(self):
        """
        Returns the rotation between the AR marker and the object in quaternion form
        """
        return tf.transformations.quaternion_from_matrix(self.T_ar_obj.matrix)

    @property
    def t_ar_obj(self):
        """
        Returns the translation between the AR marker and the object
        """
        return self.T_ar_obj.translation

OBJECT_TEMPLATES = {
    #ObjectTemplate(name='spray', ar_marker='ar_marker_8', t_ar_obj=[-0.089, -0.066, 0.106]),
    ObjectTemplate(name='gearbox', ar_marker='ar_marker_1', t_ar_obj=[-0.067, -0.06195, 0.056]),
    # ObjectTemplate(name='gearbox', ar_marker='ar_marker_9', t_ar_obj=[-0.11, 0.06, 0.056]),
    #ObjectTemplate(name='mount2', ar_marker='ar_marker_10', t_ar_obj=[-0.103, -0.064, 0.038])
}
'''

if __name__ == '__main__':
	rospy.init_node('sim_publisher')
	print('started node sim_publisher')
	broadcaster = tf.TransformBroadcaster()
	listener = tf.TransformListener()
	print('initialized broadcaster and listener')
	trans = (0.5, 0, 0) # translation
	rot = (0, 0, 0, 1) # rotation as quaternion
	sim_ar_name = 'ar_marker_1' # tf to publish

	rate = rospy.Rate(1.0)
	while not rospy.is_shutdown():
		#object_template = OBJECT_TEMPLATES[0] # gearbox only
		try:
			broadcaster.sendTransform(
				trans,
				rot,
				rospy.Time.now(),
				sim_ar_name,
				'base'
			)
		except:
			continue
	rate.sleep()

