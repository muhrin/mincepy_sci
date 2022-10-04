# -*- coding: utf-8 -*-
import pytest

pandas = pytest.importorskip("pandas")


def test_basic(historian):
    frame_dict = {"col1": [1, 2], "col2": [3, 4]}
    frame = pandas.DataFrame(data=frame_dict)

    frame_id = historian.save(frame)
    del frame

    frame = historian.load(frame_id)
    assert frame.to_dict("list") == frame_dict
