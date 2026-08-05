#!/usr/bin/env python3
"""Ensure the generated live-service Python extensions match ROS Humble."""

from rclpy.type_support import check_is_valid_srv_type
from roomie.srv import MutateScene, QueryScene


def test_query_scene_python_typesupport_is_loadable():
    # Importing the generated Python class alone does not load its native
    # extension. This is the same check rclpy performs in create_client().
    check_is_valid_srv_type(QueryScene)
    assert QueryScene.__class__._TYPE_SUPPORT is not None
    assert QueryScene.Request.__class__._TYPE_SUPPORT is not None
    assert QueryScene.Response.__class__._TYPE_SUPPORT is not None


def test_mutate_scene_python_typesupport_is_loadable():
    check_is_valid_srv_type(MutateScene)
    assert MutateScene.__class__._TYPE_SUPPORT is not None
    assert MutateScene.Request.__class__._TYPE_SUPPORT is not None
    assert MutateScene.Response.__class__._TYPE_SUPPORT is not None
