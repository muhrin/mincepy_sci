# -*- coding: utf-8 -*-

import uuid

import mincepy
import numpy


class ArrayHelper(mincepy.TypeHelper):
    TYPE = numpy.ndarray
    TYPE_ID = uuid.UUID("eff7de75-2d6c-48dd-9b46-0ce16fb8b688")

    def yield_hashables(
        self, array: numpy.ndarray, hasher
    ):  # pylint: disable=arguments-differ
        yield from hasher.yield_hashables(array.tolist())

    def eq(self, one, other) -> bool:
        return (one == other).all()

    def save_instance_state(
        self, array: numpy.ndarray, _referencer
    ):  # pylint: disable=arguments-differ
        return array.tolist()

    def new(self, encoded_saved_state):
        return numpy.asarray(encoded_saved_state)

    def load_instance_state(
        self, array: numpy.ndarray, saved_state, _referencer
    ):  # pylint: disable=arguments-differ
        pass  # Nothing to do, did it all in new


TYPES = (ArrayHelper(),)
