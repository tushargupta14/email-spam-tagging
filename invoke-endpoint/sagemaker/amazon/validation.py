# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Placeholder docstring"""
from __future__ import absolute_import


def gt(minimum):
    """
    Args:
        minimum:
    """

    def validate(value):
        return value > minimum

    return validate


def ge(minimum):
    """
    Args:
        minimum:
    """

    def validate(value):
        return value >= minimum

    return validate


def lt(maximum):
    """
    Args:
        maximum:
    """

    def validate(value):
        return value < maximum

    return validate


def le(maximum):
    """
    Args:
        maximum:
    """

    def validate(value):
        return value <= maximum

    return validate


def isin(*expected):
    """
    Args:
        *expected:
    """

    def validate(value):
        return value in expected

    return validate


def istype(expected):
    """
    Args:
        expected:
    """

    def validate(value):
        return isinstance(value, expected)

    return validate
