# -*- coding: utf-8 -*-
"""
test main entrance
"""
import unittest

suite = unittest.TestLoader().discover("./bil/generate/tests")
unittest.TextTestRunner().run(suite)
