from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import add_datavar, add_time


@add_datavar(name="waterlevel", default_value=0.0)
@add_time(grid_coord=True)
class WaterLevel(GriddedSkeleton):
    pass
