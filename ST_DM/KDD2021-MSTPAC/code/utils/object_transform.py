#############################################################################
 # 
 # Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
 # 
############################################################################
"""
 Specify the brief object_transform.py
 Author: map(wushilei@baidu.com)
 Date: 2019/07/15 16:37:12
"""

import pickle 
import base64 

class ObjectTransform(object):
    """
    transform object and string
    """
    @classmethod
    def pickle_dumps_to_str(cls, obj):
        """
        from object to str
        """
        try:
            #return base64.encodebytes(pickle.dumps(obj)).decode()
            #return base64.b64encode(pickle.dumps(obj))
            return base64.b64encode(pickle.dumps(obj)).decode()
        except pickle.PicklingError:
            pass

    @classmethod
    def pickle_loads_from_str(cls, obj_str):
        """
        from str to object
        """
        try:
            #return pickle.loads(base64.decodebytes(obj_str.encode()))
            #return pickle.loads(base64.b64decode(obj_str))
            return pickle.loads(base64.b64decode(obj_str.encode()))
        except pickle.UnpicklingError:
            pass

