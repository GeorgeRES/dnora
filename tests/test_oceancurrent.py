from dnora.grid import Grid
from dnora import modelrun, current
import numpy as np

from dnora.readers.generic_readers import ConstantGriddedData


def test_import_constant_current_one_point():
    grid = Grid(lon=5, lat=60)
    model = modelrun.ModelRun(
        grid, start_time="2020-01-01 00:00", end_time="2020-01-02 00:00"
    )

    model.import_current(ConstantGriddedData(), u=1, v=2)

    assert model.current().size() == (25, 1, 1)

    np.testing.assert_almost_equal(np.mean(model.current().u()), 1)
    np.testing.assert_almost_equal(np.mean(model.current().v()), 2)
    np.testing.assert_almost_equal(
        np.mean(model.current().magnitude()), (2**2 + 1**2) ** 0.5
    )

    np.testing.assert_almost_equal(
        np.mean(model.current().direction()),
        90 - np.rad2deg(np.arctan2(2, 1)) + 180,
    )


def test_import_constant_current():
    grid = Grid(lon=(5, 6), lat=(60, 61))
    grid.set_spacing(nx=5, ny=10)

    model = modelrun.ModelRun(
        grid, start_time="2020-01-01 00:00", end_time="2020-01-02 00:00"
    )

    model.import_current(ConstantGriddedData(), u=1, v=2)

    assert model.current().size() == (25, 10, 5)

    np.testing.assert_almost_equal(np.mean(model.current().u()), 1)
    np.testing.assert_almost_equal(np.mean(model.current().v()), 2)
    np.testing.assert_almost_equal(
        np.mean(model.current().magnitude()), (2**2 + 1**2) ** 0.5
    )