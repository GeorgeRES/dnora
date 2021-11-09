from abc import ABC, abstractmethod
import xarray as xr
import numpy as np
from copy import copy
from scipy.interpolate import griddata
import sys
import matplotlib.pyplot as plt
from .. import msg
from ..aux import distance_2points, day_list



class TopoReader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, lon_min, lon_max, lat_min, lat_max):
        pass

class EmptyTopo(TopoReader):
    """Creates an empty topography. Called when setting initial spacing."""
    def __init__(self, grid):
        self.grid = copy(grid)
        pass

    def __call__(self, lon_min, lon_max, lat_min, lat_max):
        # Creates a trivial topography with all water points
        topo = np.ones((self.grid.data.ny,self.grid.data.nx))*-9999
        topo_lon = copy(self.grid.lon())
        topo_lat = copy(self.grid.lat())
        return topo, topo_lon, topo_lat


class GridProcessor(ABC):
    def __init__(self):
        pass

    def __call__(self, data, lon, lat, land_sea_mask, boundary_mask):
        pass

class TrivialFilter(GridProcessor):
    def __init__(self):
        pass

    def __call__(self, data, lon, lat, land_sea_mask, boundary_mask):
        return copy(data)


class Mesher(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data, lon, lat, lonQ, latQ):
        pass


class Interpolate(Mesher):
    def __init__(self, method = 'linear'):
        self.method = method
        #msg.info(f"Initializing mesher with method: {self.method}")

        return

    def __call__(self, data, lon, lat, lonQ, latQ):
        lon0, lat0 = np.meshgrid(lon, lat)
        lon1, lat1 = np.meshgrid(lonQ, latQ)
        data[np.isnan(data)] = 0 # Keeping land points as nan lets the shoreline creep out
        M = np.column_stack((data.ravel(), lon0.ravel(),lat0.ravel()))
        meshed_data = griddata(M[:,1:], M[:,0], (lon1, lat1), method=self.method)
        meshed_data[meshed_data>=0] = 32767

        return meshed_data

class TrivialMesher(Mesher):
    def __init__(self):
        pass

    def __call__(self, data, lon, lat, lonQ, latQ):
        return copy(data)


class TopoWriter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, grid):
        pass




class Grid:
    def __init__(self, lon_min = 0, lon_max = 0, lat_min = 0, lat_max = 0, name="AnonymousGrid"):

        data_dict = {'lon_min': lon_min, 'lon_max': lon_max, 'lat_min': lat_min, 'lat_max': lat_max, 'name': name}

        # Initializes new dataset
        self.data = xr.Dataset(
                    attrs=(data_dict
                    ),
                    )
        return

    def import_topo(self, topo_fetcher: TopoReader):
        """Reads to topography"""
        msg.header(f'Importing topography with {type(topo_fetcher).__name__}')
        topo, lon, lat = topo_fetcher(self.data.lon_min, self.data.lon_max, self.data.lat_min, self.data.lat_max)


        # Initialize new dataset
        coords_dict = {'lon': lon, 'lat': lat}
        vars_dict = {'topo': (['lat', 'lon'], topo)}
        self.rawdata = xr.Dataset(
                    coords=(coords_dict
                    ),
                    data_vars=(vars_dict
                    ),
                    )

    def filter_topo(self, filt: GridProcessor = TrivialFilter()):
            msg.header(f'Filtering topography with {type(filt).__name__}')

            empty_mask = np.full(self.raw_topo().shape, False)
            land_sea_mask = self.raw_topo() < 0 # Sea points set to true

            topo = filt(self.raw_topo(), self.raw_lon(), self.raw_lat(), land_sea_mask, empty_mask)

            vars_dict = {'topo': (['lat', 'lon'], topo)}
            self.rawdata = self.rawdata.assign(vars_dict)

            return

    def mesh_grid(self, mesher: Mesher = Interpolate(method = 'linear')):
        if hasattr(self, 'lon') and hasattr(self, 'lon'):

            msg.header(f'Meshing grid bathymetry with {type(mesher).__name__}')
            topo = mesher(self.raw_topo(), self.raw_lon(), self.raw_lat(), self.lon(), self.lat())
            vars_dict = {'topo': (['lat', 'lon'], topo)}
            self.data = self.data.assign(vars_dict)

            self.update_masks()


        else:
            msg.templates('no_spacing')
            return


    def filter_grid(self, filt: GridProcessor = TrivialFilter()):
            msg.header(f'Filtering meshed grid with {type(filt).__name__}')
            topo = filt(self.topo(), self.lon(), self.lat(), self.land_sea_mask(), self.boundary_mask())

            vars_dict = {'topo': (['lat', 'lon'], topo)}
            self.data = self.data.assign(vars_dict)

            msg.info('Upodating land-sea mask and boundary mask')
            self.update_masks()

    def update_masks(self):
        self.set_land_sea_mask()

        # Create empty (no boundary points) if doesn't exist
        if self.boundary_mask().size == 0:
            self.set_boundary(bounN = 0)
        return

    def drop_topo_and_masks(self):
        if self.topo().size > 0:
            self.data =self.data.drop('topo')
        if self.land_sea_mask().size > 0:
            self.data =self.data.drop('land_sea_mask')
        if self.boundary_mask().size > 0:
            self.data =self.data.drop('boundary_mask')
        return

    def set_land_sea_mask(self, given_array = None):
        if given_array is None:
            land_sea_mask = np.full(self.topo().shape, False)
            land_sea_mask = self.topo() < 0 # Sea points set to true
        else:
            land_sea_mask = given_array

        vars_dict = {'land_sea_mask': (['lat', 'lon'], land_sea_mask)}
        self.data = self.data.assign(vars_dict)

        return

    def set_boundary(self, bounN = -1, edges = ['N', 'S', 'E', 'W'], given_array = None):
        """Define boundary points in grid either by setting every bounN point at edges, or providins a boolean mask (array)."""

        if (given_array is None) and bounN < 0:
            msg.advice('Give either bounN or given_array')
            return

        msg.header("Setting boundary points")
        if given_array is not None: # Array where boundary points are True and rest False is given
            boundary_mask = given_array
        elif bounN >= 0:
            bounN = int(bounN)

            if bounN >0:
                boundary_mask = np.full(self.land_sea_mask().shape, False)
                msg.info(f"Setting boundary points using bounN = {bounN}")
                # --------- North boundary ----------
                if 'N' in edges:
                    boundary_mask[-1,::bounN] = True
                ## --------- South boundary ----------
                if 'S' in edges:
                    boundary_mask[0,::bounN] = True
                ## --------- East boundary ----------
                if 'E' in edges:
                    boundary_mask[::bounN,-1] = True
                ## --------- West boundary ----------
                if 'W' in edges:
                    boundary_mask[::bounN,0] = True
                msg.plain(f'Set {sum(sum(boundary_mask)):d} boundary points (some can be on land)')
            elif bounN == 0:
                if self.boundary_mask().size > 0:
                    msg.info("Removing all boundary points")
                boundary_mask = np.full(self.land_sea_mask().shape, False)


        vars_dict = {'boundary_mask': (['lat', 'lon'], boundary_mask)}
        self.data = self.data.assign(vars_dict)
        #print(self)
        return




    def point_list(self, mask):
        meshlon, meshlat=np.meshgrid(self.lon(),self.lat())
        lonlat_flat = np.column_stack((meshlon.ravel(),meshlat.ravel()))
        mask_flat = mask.ravel()

        return lonlat_flat[mask_flat]

    def boundary_points(self):
        """Returns a lon, lat list of the set boundary points."""
        if self.boundary_mask().size > 0:
            mask = np.logical_and(self.boundary_mask(), self.land_sea_mask())
            BOUND = self.point_list(mask)
            return BOUND
        else:
            return np.array([])

    def land_points(self):
        """Returns a lon, lat list of land points."""
        if self.boundary_mask().size > 0:
            mask = np.logical_not(self.land_sea_mask())
            LAND = self.point_list(mask)
            return LAND
        else:
            return np.array([])

    def sea_points(self):
        """Returns a lon, lat list of land points."""
        if self.boundary_mask().size > 0:
            mask = self.land_sea_mask()
            LAND = self.point_list(mask)
            return LAND
        else:
            return np.array([])

    def land_sea_mask(self):
        if hasattr(self.data, 'land_sea_mask'):
            return copy(self.data.land_sea_mask.values)
        else:
            return np.array([])
    def boundary_mask(self):
        if hasattr(self.data, 'boundary_mask'):
            return copy(self.data.boundary_mask.values)
        else:
            return np.array([])
    def raw_topo(self):
        return copy(self.rawdata.topo.values)

    def raw_lon(self):
        return copy(self.rawdata.lon.values)

    def raw_lat(self):
        return copy(self.rawdata.lat.values)

    def topo(self):
        if hasattr(self.data, 'topo'):
            return copy(self.data.topo.values)
        else:
            return np.array([])

    def nx(self):
        if hasattr(self.data, 'nx'):
            return copy(self.data.nx)
        else:
            return 0

    def ny(self):
        if hasattr(self.data, 'ny'):
            return copy(self.data.ny)
        else:
            return 0

    def lon(self):
        if hasattr(self.data, 'lon'):
            lon = copy(self.data.lon.values)
        else:
            lon = np.array([self.data.lon_min, self.data.lon_max])
        return lon

    def lat(self):
        if hasattr(self.data, 'lat'):
            lat = copy(self.data.lat.values)
        else:
            lat = np.array([self.data.lat_min, self.data.lat_max])
        return lat

    def name(self):
        return copy(self.data.name)

    def set_spacing(self, dlon = 0, dlat = 0, dm = 0, nx = 0, ny = 0, floating_edge = False):
        """Defines longitude and latitude vectors based on desired spacing. Use either dlon and dlat (in deg), dm (in m), or nx and ny (in grid points)."""
        msg.header("Setting grid spacing")

        dont_update = False
        if dlon and dlat:
            if floating_edge:
                #Use exactly given dlon/dlat and change lon_max/lat_max accordingly

                msg.plain(f"Setting spacing based on dlon = {dlon} and dlat = {dlat}")
                msg.plain("floating_edge = True. Making sure dlon/dlat are keep exactly fixed")

                old_lat_max=self.lat()[-1]
                old_lon_max=self.lon()[-1]
                lon=np.arange(self.data.lon_min,self.data.lon_max+dlon/2,dlon)
                lat=np.arange(self.data.lat_min,self.data.lat_max+dlon/2,dlat)

                msg.plain(f"Setting lon_max ({old_lon_max} >> {self.lon()[-1]}), lat_max ({old_lat_max} >> {self.lat()[-1]})")
                lon_max = lon[-1]
                lat_max = lat[-1]
                lon_min = lon[0]
                lat_min = lat[0]

                distance_x = distance_2points((lat_min+lat_max)/2, lon_min, (lat_min+lat_max)/2, lon_max)
                distance_y = distance_2points(lat_min, lon_min, lat_max, lon_min)

                # Number of points
                #nx = int((lon_max-self.data.lon_min)/dlon + 1)
                #ny = int((lat_max-self.data.lat_min)/dlat + 1)
                nx = len(lon)
                ny = len(lat)

                # dx, dy in metres
                dx = distance_x*1000/nx
                dy = distance_y*1000/ny


                attr_dict = {'lon_max': lon_max, 'lat_max': lat_max}
                self.data = self.data.assign_attrs(attr_dict)



            else:
                # Keeping edges fixed and rounding dlon/dlat to something suitable

                msg.plain(f"Setting spacing based on dlon = {dlon} and dlat = {dlat}")

                # Number of points
                nx = int((self.data.lon_max-self.data.lon_min)/dlon + 1)
                ny = int((self.data.lat_max-self.data.lat_min)/dlat + 1)

                # Define longitudes and latitudes
                lon = np.linspace(self.data.lon_min, self.data.lon_max, nx)
                lat = np.linspace(self.data.lat_min, self.data.lat_max, ny)
                dlon = (self.data.lon_max-self.data.lon_min)/(nx-1)
                dlat = (self.data.lat_max-self.data.lat_min)/(ny-1)

                distance_x = distance_2points((self.data.lat_min+self.data.lat_max)/2, self.data.lon_min, (self.data.lat_min+self.data.lat_max)/2, self.data.lon_max)
                distance_y = distance_2points(self.data.lat_min, self.data.lon_min, self.data.lat_max, self.data.lon_min)

                # dx, dy in metres
                dx = distance_x*1000/nx
                dy = distance_y*1000/ny


            if dm:
                msg.plain(f"Ignoring value dm={dm} metres!")
        elif dm:
            msg.plain(f"Setting spacing based on (approximately) dm={dm} metres")
            distance_x = distance_2points((self.data.lat_min+self.data.lat_max)/2, self.data.lon_min, (self.data.lat_min+self.data.lat_max)/2, self.data.lon_max)
            distance_y = distance_2points(self.data.lat_min, self.data.lon_min, self.data.lat_max, self.data.lon_min)

            # Number of points
            nx = int(np.round(distance_x*1000/dm))
            ny = int(np.round(distance_y*1000/dm))

            # dx, dy in metres
            dx = distance_x*1000/nx
            dy = distance_y*1000/ny


            # Define longitudes and latitudes
            lon = np.linspace(self.data.lon_min, self.data.lon_max, nx)
            lat = np.linspace(self.data.lat_min, self.data.lat_max, ny)
            dlon = (self.data.lon_max-self.data.lon_min)/(nx-1)
            dlat = (self.data.lat_max-self.data.lat_min)/(ny-1)



        elif nx and ny:
            msg.plain(f"Setting spacing to have nx = {nx}, ny = {ny} points.")

            # Define longitudes and latitudes
            lon = np.linspace(self.data.lon_min, self.data.lon_max, nx)
            lat = np.linspace(self.data.lat_min, self.data.lat_max, ny)
            dlon = (self.data.lon_max-self.data.lon_min)/(nx-1)
            dlat = (self.data.lat_max-self.data.lat_min)/(ny-1)

            distance_x = distance_2points((self.data.lat_min+self.data.lat_max)/2, self.data.lon_min, (self.data.lat_min+self.data.lat_max)/2, self.data.lon_max)
            distance_y = distance_2points(self.data.lat_min, self.data.lon_min, self.data.lat_max, self.data.lon_min)

            # dx, dy in metres
            dx = distance_x*1000/nx
            dy = distance_y*1000/ny


        else:
            dont_update = True


        if dont_update:
            msg.advice("Doing nothing. Run set_spacing with either dlon AND dlat (in degrees), nx AND ny (in grid points), or dm (in metres).")
        else:

            # Old topography conflicts in size, so drop them first
            self.drop_topo_and_masks()

            attr_dict = {'nx': nx, 'ny': ny, 'dx': dx, 'dy': dy, 'dlon': dlon, 'dlat': dlat}
            self.data = self.data.assign_attrs(attr_dict)

            coords_dict = {'lon': lon, 'lat': lat}
            self.data = self.data.assign_coords(coords_dict)


        # Initialize the grid with an empty topography
        msg.info("Initializing with an empty topography")
        topo_fetcher = EmptyTopo(self)
        self.import_topo(topo_fetcher)
        #self.filter_topo(filt = TrivialFilter())
        self.mesh_grid(mesher = TrivialMesher())

        print(self)
    def plot(self, save_fig = False, filename = '', boundary = None, grid_processor: GridProcessor = TrivialFilter()):
        """Creates a plot of the topography"""

        topo = grid_processor(self.topo(), self.lon(), self.lat(), self.land_sea_mask(), self.boundary_mask())

        if self.topo().size > 0:
            if not filename:
                filename = f"{self.data.name}_topo.pdf"
            levels = np.linspace(np.min(topo), 0, 100, endpoint=True)

            plt.figure()
            plt.contourf(self.lon(),self.lat(),topo,levels)

            if boundary is not None:
                plt.plot(boundary.lon(), boundary.lat(),'kx')

            plt.colorbar()
            plt.title(f"{self.data.name} topography")
            plt.show()

            if save_fig:
                plt.savefig(filename, dpi=300)
                msg.to_file(filename)
        else:
            msg.templates('no_topo')


    def plot_mask(self, save_fig = False, filename = '', boundary = None):
        """Creates a plot of the land-sea mask"""
        if self.boundary_mask().size > 0:
            if not filename:
                filename = f"{self.data.name}_mask.pdf"

            plt.figure()
            plt.contourf(self.lon(),self.lat(),self.land_sea_mask())
            BOUND = self.boundary_points()
            plt.plot(BOUND[:,0], BOUND[:,1],'r*')

            if boundary is not None:
                plt.plot(boundary.lon(), boundary.lat(),'kx')


            plt.colorbar()
            plt.title(f"{self.data.name} land-sea mask")
            #plt.show()

            if save_fig:
                plt.savefig(filename, dpi=300)
                msg.to_file(filename)
        else:
            msg.templates('no_mask')

    def write_status(self, filename = '', folder = ''):
        """Writes out the status of the grid to a file."""
        if not filename:
            filename = f"{self.data.name}_info.txt"

        filename = folder + filename
        msg.to_file(filename)

        stdout = sys.stdout
        sys.stdout = open(filename, 'w')
        print(self)
        sys.stdout.close()
        sys.stdout = stdout


    def __str__(self):
            msg.header(f"Status of grid {self.data.name}")
            msg.plain(f'lon: {self.data.lon_min} - {self.data.lon_max}, lat: {self.data.lat_min} - {self.data.lat_max}')
            if hasattr(self.data, 'dlon') and hasattr(self.data, 'dlon'):
                msg.plain(f'dlon, dlat = {self.data.dlon}, {self.data.dlat} deg')
                msg.plain(f'Inverse: dlon, dlat = 1/{1/self.data.dlon}, 1/{1/self.data.dlat} deg')
            if hasattr(self.data, 'dx') and hasattr(self.data, 'dy'):
                msg.plain(f'dx, dy approximately {self.data.dx}, {self.data.dy} metres')
            if hasattr(self.data, 'nx') and hasattr(self.data, 'ny'):
                msg.plain(f'nx, ny = {self.data.nx} x {self.data.ny} grid points')
            if self.topo().size > 0:
                msg.plain(f"Mean depth: {np.mean(self.topo()[self.land_sea_mask()]):.1f} m")
                msg.plain(f"Max depth: {np.min(self.topo()[self.land_sea_mask()]):.1f} m")
                msg.plain(f"Min depth: {np.max(self.topo()[self.land_sea_mask()]):.1f} m")
            if self.land_sea_mask().size > 0:
                msg.print_line()
                msg.plain('Grid contains:')
                msg.plain(f'{sum(sum(self.land_sea_mask())):d} sea points')
                msg.plain(f'{sum(sum(np.logical_not(self.land_sea_mask()))):d} land points')
            if self.boundary_mask().size > 0:
                msg.plain(f'{sum(sum(np.logical_and(self.boundary_mask(), self.land_sea_mask()))):d} boundary points')
            msg.print_line()
            return ''


