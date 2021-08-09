import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
import netCDF4
from dnora2 import msg
from dnora2 import spec
from copy import copy

# -----------------------------------------------------------------------------
# MISC STAND ALONE FUNCTIONS
# -----------------------------------------------------------------------------
def distance_2points(lat1,lon1,lat2,lon2):
    """Calculate distance between two points"""
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c # in km
    return distance


def min_distance(lon, lat, lon_vec, lat_vec):
    dx = []
    for n in range(len(lat_vec)):
        dx.append(distance_2points(lat, lon, lat_vec[n], lon_vec[n]))
        
    return np.array(dx).min(), np.array(dx).argmin()
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# POINT PICKER CLASSES FOR DIFFERENT METHODS TO CHOOSE SPECTRA FROM DATABASE
# -----------------------------------------------------------------------------
class PointPicker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, grid, bnd_lon, bnd_lat):
        return 


class TrivialPicker(PointPicker):
    def __init__(self):
        pass

    def __call__(self, grid, bnd_lon, bnd_lat):
        inds = np.array(range(len(bnd_lon)))
        return inds


class NearestGridPointPicker(PointPicker):
    def __init__(self):
        pass
    
    def __call__(self, grid, bnd_lon, bnd_lat):
        bnd_points = grid.bnd_points()
        lon = bnd_points[:,0]
        lat = bnd_points[:,1]
        
        # Go through all points where we want output and find the nearest available point
        inds = []
        for n in range(len(lat)):
            dx, ind = min_distance(lon[n], lat[n], bnd_lon, bnd_lat)
            msg.plain(f"Point {n}: lat: {lat[n]:10.7f}, lon: {lon[n]:10.7f} <<< ({bnd_lat[ind]: .7f}, {bnd_lon[ind]: .7f}). Distance: {dx:.1f} km")
            inds.append(ind) 

        inds = np.array(inds)
        return inds
    
class AreaPicker(PointPicker):
    def __init__(self, expansion_factor = 1.5):
        self.expansion_factor = expansion_factor
        return

    def __call__(self, grid, bnd_lon, bnd_lat):
        
        # Define area to search in
        expand_lon = (grid.lon_max - grid.lon_min)*self.expansion_factor*0.5
        expand_lat = (grid.lat_max - grid.lat_min)*self.expansion_factor*0.5
        
        # Get all the spectra in this area
        lon0=grid.lon_min - expand_lon
        lon1=grid.lon_max + expand_lon
        
        lat0=grid.lat_min - expand_lat
        lat1=grid.lat_max + expand_lat

        masklon = np.logical_and(bnd_lon < lon1, bnd_lon > lon0)
        masklat = np.logical_and(bnd_lat > lat0, bnd_lat < lat1)
        mask=np.logical_and(masklon, masklat)
        
        inds = np.where(mask)[0]
        
        msg.info(f"Found {len(inds)} points inside {lon0:10.7f}-{lon1:10.7f}, {lat0:10.7f}-{lat1:10.7f}.")

        return inds
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# BOUNDARY FETCHER CLASSES RESPONSIBLE FOR ACTUALLY READING THE SPECTRA
# -----------------------------------------------------------------------------

class BoundaryFetcher(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, start_time, end_time):
        pass

    def days(self):
        """Determins a Pandas data range of all the days in the time span of the InputModel objext"""
        days = pd.date_range(start=self.start_time.split('T')[0], end=self.end_time.split('T')[0], freq='D')
        return days

    def get_time_limits(self, ind):
        """Determines star and end time for the day. First and last day doesn't start at 00:00 or end at 23:59"""
        if ind == 0:
            t0 = self.start_time
            t1 = self.days()[0].strftime('%Y-%m-%d') + "T23:59:59"
        elif ind == (len(self.days())-1):
            t0 = self.days()[-1].strftime('%Y-%m-%d') + "T00:00:00"				
            t1 = self.end_time
        else:
            t0 = self.days()[ind].strftime('%Y-%m-%d') + "T00:00:00"	
            t1 = self.days()[ind].strftime('%Y-%m-%d') + "T23:59:59"
        return t0, t1

    def __str__(self):
        return (f"{self.start_time} - {self.end_time}")


class BoundaryWAM4(BoundaryFetcher):
    
    def get_coordinates(self, start_time):
        """Reads first time instance of first file to get longitudes and latitudes for the PointPicker"""
        day = pd.date_range(start_time, start_time,freq='D')
        url = self.get_url(day[0])

        data = xr.open_dataset(url).isel(time = [0])
        
        lon_all = data.longitude.values[0]
        lat_all = data.latitude.values[0]
        
        return lon_all, lat_all
    
    def __call__(self, start_time, end_time, inds):
        """Reads in all boundary spectra between the given times and at for the given indeces"""
        self.start_time = start_time
        self.end_time = end_time
        
        msg.info(f"Getting boundary spectra from WAM4 from {self.start_time} to {self.end_time}")
        bnd_list = []    
        for n in range(len(self.days())):
            url = self.get_url(self.days()[n])
            msg.plain(url)
            t0, t1 = self.get_time_limits(n)
            bnd_list.append(xr.open_dataset(url).sel(time = slice(t0, t1), x = (inds+1)))
            
        bnd=xr.concat(bnd_list, dim="time").squeeze('y')
        
        time = bnd.time.values
        freq = bnd.freq.values
        dirs = bnd.direction.values
        spec = bnd.SPEC.values
        lon = bnd.longitude.values
        lat = bnd.latitude.values
        
        source = f"{bnd.title}, {bnd.institution}"
        
        return  time, freq, dirs, spec, lon, lat, source

    def get_url(self, day):
        url = 'https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam4archive/'+day.strftime('%Y') +'/'+day.strftime('%m')+'/'+day.strftime('%d')+'/MyWave_wam4_SPC_'+day.strftime('%Y%m%d')+'T00Z.nc'
        return url
    

class BoundaryNORA3(BoundaryFetcher):
    
    def get_coordinates(self, start_time):
        """Reads first time instance of first file to get longitudes and latitudes for the PointPicker"""
        day = pd.date_range(start_time, start_time,freq='D')
        url = self.get_url(day[0])

        data = xr.open_dataset(url).isel(time = [0])
        
        lon_all = data.longitude.values[0]
        lat_all = data.latitude.values[0]
        
        return lon_all, lat_all
    
    def __call__(self, start_time, end_time, inds):
        """Reads in all boundary spectra between the given times and at for the given indeces"""
        self.start_time = start_time
        self.end_time = end_time
        
        msg.info(f"Getting boundary spectra from NORA3 from {self.start_time} to {self.end_time}")
        bnd_list = []    
        for n in range(len(self.days())):
            url = self.get_url(self.days()[n])
            msg.plain(url)
            t0, t1 = self.get_time_limits(n)
            bnd_list.append(xr.open_dataset(url).sel(time = slice(t0, t1), x = (inds+1)))
            
        bnd=xr.concat(bnd_list, dim="time").squeeze('y')
        
        time = bnd.time.values
        freq = bnd.freq.values
        dirs = bnd.direction.values
        spec = bnd.SPEC.values
        lon = bnd.longitude.values[0,:]
        lat = bnd.latitude.values[0,:]
        
        source = f"{bnd.title}, {bnd.institution}"
        
        return  time, freq, dirs, spec, lon, lat, source
    

    def get_url(self, day):
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/'+day.strftime('%Y') +'/'+day.strftime('%m')+'/SPC'+day.strftime('%Y%m%d')+'00.nc'
        return url
# -----------------------------------------------------------------------------

# =============================================================================
#  BOUNDARY OBJECT CONTAINING THE ACTUAL DATA
# =============================================================================

class Boundary:
    def __init__(self, grid, name = "AnonymousBoundary"):
        self.grid = copy(grid)
        self.name = name
        return

    def import_boundary(self, start_time: str, end_time: str, boundary_fetcher: BoundaryFetcher,  point_picker: PointPicker = TrivialPicker()):
        
        lon_all, lat_all = boundary_fetcher.get_coordinates(start_time)
        
        msg.header(f"Choosing spectra with {type(point_picker).__name__} using expansion_factor = {point_picker.expansion_factor:.2f}")
        inds = point_picker(self.grid, lon_all, lat_all)

        msg.header(f"Fetching data using {type(boundary_fetcher).__name__}")
        time, freq, dirs, spec, lon, lat, source = boundary_fetcher(start_time, end_time, inds)
        self.data = self.compile_to_xr(time, freq, dirs, spec, lon, lat, source)
        
        self.mask = [True]*len(self.time())
        
        return
    
    
    def process_spectra(self, spectral_processors = spec.TrivialSpectralProcessor(calib_spec = 1)):
    
        if not isinstance(spectral_processors, list):
            spectral_processors = [spectral_processors]
            
        for n in range (len(spectral_processors)):    
            spectral_processor = spectral_processors[n]
            msg.header(f"Processing spectra with {type(spectral_processor).__name__}")
            
            new_spec, new_mask, new_freq, new_dirs = spectral_processor(self.spec(), self.freq(), self.dirs(), self.time(), self.x(), self.lon(), self.lat(), self.mask)
            
            self.data.spec.values = new_spec
            self.mask = new_mask
            
            self.data = self.data.assign_coords(dirs=new_dirs)
            self.data = self.data.assign_coords(freq=new_freq)

        return
            
    def compile_to_xr(self, time, freq, dirs, spec, lon, lat, source):
        x = np.array(range(spec.shape[1]))
        data = xr.Dataset(
            data_vars=dict(
                spec=(["time", "x", "freq", "dirs"], spec),
            ),
            coords=dict(
                freq=freq,
                dirs=dirs,
                x=x,
                lon=(["x"], lon),
                lat=(["x"], lat),
                time=time,
            ),
            attrs=dict(source=source,
                name=self.name
            ),
            )
        return data  
    
    def slice_data(self, start_time: str = '', end_time: str = '', x = []):
        if isinstance(x, int):
            x = [x]
        elif not x:
            x=self.x()

        if not start_time:
            # This is not a string, but slicing works also with this input
            start_time = self.time()[0] 
        
        if not end_time:
            # This is not a string, but slicing works also with this input
            end_time = self.time()[-1]
            
        sliced_data = self.data.sel(time=slice(start_time, end_time), x=x)
            
        return sliced_data
    
    def spec(self, start_time: str = '', end_time: str = '', x = []):
        spec = self.slice_data(start_time, end_time, x).spec.values
            
        return spec
    
    
    def time(self):
        return self.data.time.values

    def freq(self):
        return self.data.freq.values
    
    def dirs(self):
        return self.data.dirs.values
    
    def lon(self):
        return self.data.lon.values
    
    def lat(self):
        return self.data.lat.values
    
    def x(self):
        return self.data.x.values
    
    

# -----------------------------------------------------------------------------
# OUTPUT MODEL CLASSES RESPONSIBLE FOR WRITING SPECTRA IN CORRECT FORMAT
# -----------------------------------------------------------------------------
class OutputModel(ABC):
    bnd_in: list
    bnd_points: np.array
    message: str
    @abstractmethod
    def __call__(self, bnd_out):
        pass

 
class OutputWW3nc(OutputModel):
    def __init__(self):
        pass

    def __call__(self, in_boundary: Boundary):
        boundary = copy(in_boundary)
        msg.header(f"Writing output with {type(self).__name__}")
        
        # Convert from oceanic to mathematical convention
        boundary.process_spectra(spec.NautToWW3())
        
        msg.info('Writing WAVEWATCH-III netcdf-output')

        for n in range(len(boundary.x())):
            if boundary.mask[n]:
                self.write_netcdf(boundary, n)
            else:
                msg.info(f"Skipping point {n} ({boundary.lon()[n]:10.7f}, {boundary.lat()[n]:10.7f}). Masked as False.")
        return
    
    def write_netcdf(self, boundary, n):
        """Writes WW3 compatible netcdf spectral output from a list containing xarray datasets."""
        lat=boundary.lat()[n]
        lon=boundary.lon()[n]

        output_file = f"ww3_spec_E{lon:09.6f}N{lat:09.6f}.nc"
        #output_file = 'ww3_spec_E'+str(lon)+'N'+str(lat)+'.nc'
        #output_file = 'Test_ww3.nc'
        msg.plain(f"Point {n}: {output_file}")
        root_grp = netCDF4.Dataset(output_file, 'w', format='NETCDF4')
        #################### dimensions
        root_grp.createDimension('time', None)
        root_grp.createDimension('station', 1)
        root_grp.createDimension('string16', 16)
        root_grp.createDimension('frequency', len(boundary.freq()))
        root_grp.createDimension('direction', len(boundary.dirs()))
        #######################################################
        ####################### variables
        time = root_grp.createVariable('time', np.float64, ('time',))
        station = root_grp.createVariable('station', np.int32, ('station',))
        frequency = root_grp.createVariable('frequency',np.float32 , ('frequency',))
        direction = root_grp.createVariable('direction', np.float32, ('direction',))
        efth = root_grp.createVariable('efth', np.float32, ('time','station','frequency','direction',))
        latitude = root_grp.createVariable('latitude',np.float32 , ('time','station',))
        longitude = root_grp.createVariable('longitude',np.float32 , ('time','station',))
        station_name = root_grp.createVariable('station_name', 'S1', ('station','string16',))
        string16 = root_grp.createVariable('string16',np.int32 , ('string16',))
               
        ########################## Attributes
        time.units = 'seconds since 1970-01-01 00:00:00 UTC'
        time.calendar = "standard"
        time.standard_name = "time" 
        time.axis = "T" 
        
        station.long_name = "station id" 
        station.axis = "X" 
        
        frequency.units = "s-1" 
        frequency.long_name = "frequency of center band" 
        frequency.standard_name = "sea_surface_wave_frequency" 
        frequency.globwave_name = "frequency" 
        frequency.valid_min = 0 
        frequency.valid_max = 10 
        frequency.axis = "Y" 
        
        direction.units = "degree" 
        direction.long_name = "sea surface wave to direction" 
        direction.standard_name = "sea_surface_wave_to_direction" 
        direction.globwave_name = "direction" 
        direction.valid_min = 0
        direction.valid_max = 360
        direction.axis = "Z" 
        
        longitude.units='degree_east'
        longitude.long_name = "longitude" 
        longitude.standard_name = "longitude" 
        longitude.valid_min = -180
        longitude.valid_max = 180
        	#longitude:_FillValue = 9.96921e+36f ;
        longitude.content = "TX" 
        longitude.associates = "time station" 
        
        latitude.units = "degree_north" 
        latitude.long_name = "latitude" 
        latitude.standard_name = "latitude" 
        latitude.valid_min = -90
        latitude.valid_max = 90
        	#latitude:_FillValue = 9.96921e+36f ;
        latitude.content = "TX" 
        latitude.associates = "time station"
        
        station_name.long_name = "station name" 
        station_name.content = "XW" 
        station_name.associates = "station string16" 
        
        station.long_name = "station id" 
        station.axis = "X" 
        
        string16.long_name = "station_name number of characters" 
        string16.axis = "W" 
        
        efth.long_name = "sea surface wave directional variance spectral density" 
        efth.standard_name = "sea_surface_wave_directional_variance_spectral_density" 
        efth.globwave_name = "directional_variance_spectral_density" 
        efth.units = "m2 s rad-1" 
        efth.scale_factor = 1 
        efth.add_offset = 0 
        efth.valid_min = 0 
        #efth.valid_max = 1.0e+20 
        #efth._FillValue = 9.96921e+36 
        efth.content = "TXYZ" 
        efth.associates = "time station frequency direction" 
        #######################################################
        ############## Pass data
        time[:] = boundary.time().astype('datetime64[s]').astype('float64')
        frequency[:] =boundary.freq()
        direction[:] = boundary.dirs()
        
        efth[:] =  boundary.spec(x=n)
        station[:] = 1
        longitude[:] = np.full((len(boundary.time()),1), boundary.lon()[n],dtype=float)
        latitude[:] = np.full((len(boundary.time()),1), boundary.lat()[n],dtype=float)
        #longitude[:] = bnd_out.longitude.values
        #latitude[:] = bnd_out.latitude.values
        station_name[:] = 1
        
        root_grp.close() 
        return
   
   
class OutputSWANascii(OutputModel):
    def __init__(self, grid, factor = 1E-4):
        self.factor = factor
        self.grid = grid
                
    def __call__(self, bnd_in, bnd_mask):
        output_points = self.grid.bnd_points()
        # Initialize the boundary file by writing the header
        
        msg.header('Writing SWAN ASCII-output')
        
        freq, dirs = self.spec_info(bnd_in)
        with open('outfile', 'w') as file_out:
            file_out.write('SWAN   1\n')
            file_out.write('$ Data produced by '+bnd_in[0].title+'\n')
            file_out.write('TIME\n')
            file_out.write('          1\n')
            file_out.write('LONLAT\n')    
            file_out.write('          '+format(len(bnd_in))+'\n')     
            for k in range(len(bnd_in)):
                file_out.write('   '+format(output_points[k,1],'.4f')+'  '+format(output_points[k,0],'.4f')+'\n')
            file_out.write('AFREQ\n')
            file_out.write('          '+str(len(freq))+'\n')
            #breakpoint()
            for l in range(len(freq)):
                file_out.write('   '+format(freq[l],'.4f')+'\n')
            file_out.write('NDIR\n')
            file_out.write('          '+format(len(dirs))+'\n')
            for m in range(len(dirs)):
                file_out.write('   '+format(dirs[m],'.1f')+'\n') 
            file_out.write('QUANT\n')
            file_out.write('          1\n')
            file_out.write('VaDens\n')
            file_out.write('m2/Hz/degr \n')
            file_out.write('-32767\n')
                #first day
            msg.info('Generating 2d spectra at boundaries:')
    
            with open('outfile', 'w') as file_out:
                times = pd.DatetimeIndex(bnd_in[0]["time"].values) # All point have the same time vector so use first Dataset
                days = pd.date_range(start=times[0], end=times[-1], freq='D')
                
                for d in range(len(days)):
                    msg.plain(days[d].strftime('%Y-%m-%d'))
                    day_inds = np.where(times.day == days[d].day)[0]
                    
                    for time_step in day_inds:
                        time_stamp = str(times[time_step]).split('-')[0]+str(times[time_step]).split('-')[1]+\
                        str(times[time_step]).split('-')[2][:2]+'.'+str(times[time_step]).split('-')[2][3:5]+'0000\n'
                        file_out.write(time_stamp)
                        
                        for i in range(len(bnd_in)):
                            file_out.write('FACTOR\n')
                            file_out.write(format(self.factor,'1.0E')+'\n')
                            SPEC_ocean_convection = bnd_in[i].SPEC[time_step,:,:].values
                            SPEC_naut_convection = ocean_to_naut(SPEC_ocean_convection)
                            delth = 360/len(dirs)
                            np.savetxt(file_out,SPEC_naut_convection/(delth*self.factor), fmt='%-10.0f') #


        return
# -----------------------------------------------------------------------------
