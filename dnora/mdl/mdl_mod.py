from __future__ import annotations # For TYPE_CHECKING
from copy import copy
import pandas as pd
import numpy as np
from skeletons import PointSkeleton
# Import objects
from ..grd.grd_mod import Grid
from .. import grd
from ..grd.write import GridWriter

from ..file_module import FileNames
# Import abstract classes and needed instances of them
from ..wnd import Forcing
from ..wnd.read import ForcingReader
from ..wnd.write import ForcingWriter
from .. import wnd

from ..bnd import Boundary
from ..bnd.read import BoundaryReader
from ..bnd.write import BoundaryWriter
from ..bnd.pick import PointPicker
from .. import bnd


from ..spc import Spectra
from ..spc.read import SpectralReader
from ..spc.write import SpectralWriter
from .. import spc

from ..wsr import WaveSeries
from ..wsr.read import WaveSeriesReader, SpectraToWaveSeries
from ..wsr.write import WaveSeriesWriter
from .. import wsr

from ..run import ModelExecuter
from ..inp.inp import InputFileWriter
from ..spectral_grid import SpectralGrid
from skeletons.datavar_factory import add_datavar
# Import default values and aux_funcsiliry functions
from .. import msg
#from ..cacher.cache_decorator import cached_reader
from ..cacher.cache_decorator import cached_reader
from ..converters import convert_swash_mat_to_netcdf

class ModelRun:
    def __init__(self, grid: Grid=None, start_time: str='1970-01-01T00:00',
    end_time: str='2030-12-31T23:59', name: str='AnonymousModelRun',
    dry_run: bool=False):
        self.name = copy(name)
        self._grid = copy(grid)
        self._time = pd.date_range(start_time, end_time, freq='H')
        self._global_dry_run = dry_run
        self._dry_run = False  # Set by methods

    @cached_reader('Forcing', wnd.read.DnoraNc)
    def import_forcing(self, forcing_reader: ForcingReader=None,
                        name: str=None, dry_run: bool=False,
                        source: str='remote',
                        **kwargs) -> None:
        """Imports wind forcing.

        source = 'remote' (default) / '<folder>' / 'met'

        The implementation of this is up to the ForcingReader, and all options might not be functional.
        'met' options will only work in MET Norway internal networks.

        To import local netcdf files saved in DNORA format (by write_cache=True), use read_cache=True.
        """
        self._dry_run = dry_run

        forcing_reader = forcing_reader or self._get_forcing_reader()

        # This is to allow importing from cache using only a name
        if forcing_reader is None:
            raise Exception('Define a ForcingReader!')

        name = name or forcing_reader.name()
        forcing_reader.set_source(source)

        if name is None:
            raise ValueError('Provide either a name or a ForcingReader that will then define the name!')

        msg.header(forcing_reader, "Importing wind forcing...")

        if not self.dry_run():
            time, u, v, lon, lat, x, y, attributes = forcing_reader(self.grid(), self.start_time(), self.end_time(), **kwargs)
            
            self._forcing = Forcing(lon=lon, lat=lat, x=x, y=y, time=time, name=name)
            x = x or lon
            y = y or lat
            self.forcing().set_spacing(nx=len(x), ny=len(y))
            
            self.forcing().name = name 
            self.forcing().set_u(u)
            self.forcing().set_v(v)
            self.forcing().set_metadata(attributes)
        else:
            msg.info('Dry run! No forcing will be imported.')

    @cached_reader('Boundary', bnd.read.DnoraNc)
    def import_boundary(self, boundary_reader: BoundaryReader=None,
                        point_picker: PointPicker=None, name: str=None,
                        dry_run: bool=False, source: str='remote',
                       **kwargs):
        """Imports boundary spectra. Which spectra to choose spatically
        are determined by the point_picker.

        source = 'remote' (default) / '<folder>' / 'met'

        The implementation of this is up to the BoundaryReader, and all options might not be functional.
        'met' options will only work in MET Norway internal networks.
        
        To import local netcdf files saved in DNORA format (by write_cache=True), use read_cache=True.
        """

        self._dry_run = dry_run
        boundary_reader = boundary_reader or self._get_boundary_reader()
        point_picker = point_picker or self._get_point_picker()

        # This is to allow importing from cache using only a name
        if boundary_reader is None:
            raise Exception('Define a BoundaryReader!')
        if point_picker is None:
            raise Exception('Define a PointPicker!')

        name = name or boundary_reader.name()
        boundary_reader.set_source(source)

        if name is None:
            raise ValueError('Provide either a name or a BoundaryReader that will then define the name!')

      
        msg.header(boundary_reader, "Reading coordinates of spectra...")
        lon_all, lat_all, x_all, y_all = boundary_reader.get_coordinates(self.grid(), self.start_time())
        all_points = PointSkeleton(lon=lon_all, lat=lat_all, x=x_all, y=y_all)
        
        if np.all(np.logical_not(self.grid().boundary_mask())):
            boundary_points = None
        else:
            boundary_points = PointSkeleton.from_skeleton(self.grid(), mask=self.grid().boundary_mask())
        
        msg.header(point_picker, "Choosing boundary spectra...")
        inds = point_picker(self.grid(), all_points, selected_points=boundary_points, **kwargs)
        if len(inds) < 1:
            msg.warning("PointPicker didn't find any points. Aborting import of boundary.")
            return

        # Main reading happens here
        msg.header(boundary_reader, "Loading boundary spectra...")

        time, freq, dirs, spec, lon, lat, x, y, metadata = boundary_reader(self.grid(), self.start_time(), self.end_time(), inds, **kwargs)
        self._boundary = Boundary(x=x, y=y, lon=lon, lat=lat, time=time, freq=freq, dirs=dirs, name=name)

        self.boundary().set_spec(spec)
        self.boundary().set_metadata(metadata)
        # E.g. are the spectra oceanic convention etc.
        self.boundary()._convention = boundary_reader.convention()

        self.boundary().set_metadata({'spectral_convention': self.boundary().convention().value}, append=True)

        if boundary_reader.post_processing() is not None:
            self.boundary().process_boundary(boundary_reader.post_processing())
        return


    @cached_reader('Spectra', spc.read.DnoraNc)
    def import_spectra(self, spectral_reader: SpectralReader=None,
                        point_picker: PointPicker=None, name: str=None,
                        dry_run: bool=False, source: str='remote',
                       **kwargs):
        """Imports spectra. Which spectra to choose spatically
        are determined by the point_picker.

        source = 'remote' (default) / '<folder>' / 'met'

        The implementation of this is up to the SpectralReader, and all options might not be functional.
        'met' options will only work in MET Norway internal networks.
        
        To import local netcdf files saved in DNORA format (by write_cache=True), use read_cache=True.
        """

        self._dry_run = dry_run
        spectral_reader = spectral_reader or self._get_spectral_reader()
        point_picker = point_picker or self._get_point_picker()

        # This is to allow importing from cache using only a name
        if spectral_reader is None:
            raise Exception('Define a SpectralReader!')
        if point_picker is None:
            raise Exception('Define a PointPicker!')

        name = name or spectral_reader.name()
        spectral_reader.set_source(source)

        if name is None:
            raise ValueError('Provide either a name or a SpectralReader that will then define the name!')

        msg.header(spectral_reader, "Reading coordinates of spectra...")
        lon_all, lat_all, x_all, y_all = spectral_reader.get_coordinates(self.grid(), self.start_time())
        all_points = PointSkeleton(lon=lon_all, lat=lat_all, x=x_all, y=y_all)
        
        if np.all(np.logical_not(self.grid().boundary_mask())):
            boundary_points = None
        else:
            boundary_points = PointSkeleton.from_skeleton(self.grid(), mask=self.grid().boundary_mask())
        
        msg.header(point_picker, "Choosing spectra...")
        inds = point_picker(self.grid(), all_points, selected_points=boundary_points, **kwargs)

        msg.header(spectral_reader, "Loading omnidirectional spectra...")
        time, freq, spec, mdir, spr, lon, lat, x, y, metadata = spectral_reader(self.grid(), self.start_time(), self.end_time(), inds, **kwargs)

        self._spectra = Spectra(x=x, y=y, lon=lon, lat=lat, time=time, freq=freq, name=name)

        self.spectra().set_spec(spec)
        self.spectra().set_mdir(mdir)
        self.spectra().set_spr(spr)
        
        self.spectra().set_metadata(metadata)

        # E.g. are the spectra oceanic convention etc.
        self.spectra()._convention = spectral_reader.convention()
        self.spectra().set_metadata({'spectral_convention': self.spectra().convention().value}, append=True)
   
    @cached_reader('WaveSeries', wsr.read.DnoraNc)
    def import_waveseries(self, waveseries_reader: WaveSeriesReader=None,
                        point_picker: PointPicker=None, name: str=None,
                        dry_run: bool=False, source: str='remote',
                       **kwargs):


        self._dry_run = dry_run
        waveseries_reader = waveseries_reader or self._get_waveseries_reader()
        point_picker = point_picker or self._get_point_picker()

        # This is to allow importing from cache using only a name
        if waveseries_reader is None:
            raise Exception('Define a WaveSeriesReader!')
        if point_picker is None:
            raise Exception('Define a PointPicker!')

        name = name or waveseries_reader.name()
        waveseries_reader.set_source(source)

        if name is None:
            raise ValueError('Provide either a name or a WaveSeriesReader that will then define the name!')

        msg.header(waveseries_reader, "Reading coordinates of WaveSeries...")
        lon_all, lat_all, x_all, y_all = waveseries_reader.get_coordinates(self.grid(), self.start_time())

        all_points = PointSkeleton(lon=lon_all, lat=lat_all, x=x_all, y=y_all)
        
        if np.all(np.logical_not(self.grid().boundary_mask())):
            boundary_points = None
        else:
            boundary_points = PointSkeleton.from_skeleton(self.grid(), mask=self.grid().boundary_mask())
        
        msg.header(point_picker, "Choosing wave series points...")
        inds = point_picker(self.grid(), all_points, selected_points=boundary_points, **kwargs)

        msg.header(waveseries_reader, "Loading wave series data...")
        time, data_dict, lon, lat, x, y, metadata = waveseries_reader(self.grid(), self.start_time(), self.end_time(), inds, **kwargs)

        self._waveseries = WaveSeries(x, y, lon, lat, time=time, name=name)

        for wp, data in data_dict.items():
            self._waveseries = add_datavar(wp.name(), append=True)(self.waveseries()) # Creates .hs() etc. methods
            self.waveseries()._update_datavar(wp.name(), data)
            self.waveseries().set_metadata({'name': wp.name(), 'unit': f'{wp.unit()}', 'standard_name': wp.standard_name()}, data_array_name=wp.name())
                       
        self.waveseries().set_metadata(metadata) # Global attributes

    def cache_boundary(self):
        """Writes existing data to cached files."""
        self.export_boundary(boundary_writer=bnd.write.DnoraNc(), format='Cache')

    def cache_spectra(self):
        """Writes existing data to cached files."""
        self.export_spectra(spectral_writer=spc.write.DnoraNc(), format='Cache')

    def cache_forcing(self):
        """Writes existing data to cached files."""
        self.export_forcing(forcing_writer=wnd.write.DnoraNc(), format='Cache')

    def cache_waveseries(self):
        """Writes existing data to cached files."""
        self.export_waveseries(waveseries_writer=wsr.write.DnoraNc(), format='Cache')

    def boundary_to_spectra(self, dry_run: bool=False, name :str=None,
                            write_cache=False, **kwargs):
        self._dry_run = dry_run
        if self.boundary() is None:
            msg.warning('No Boundary to convert to Spectra!')

        spectral_reader = spc.read.BoundaryToSpectra(self.boundary())
        msg.header(spectral_reader, 'Converting the boundary spectra to omnidirectional spectra...')
        name = self.boundary().name

        if not self.dry_run():
            self.import_spectra(spectral_reader=spectral_reader,
                                point_picker=bnd.pick.TrivialPicker(),
                                name=name,
                                write_cache=write_cache, **kwargs)
        else:
            msg.info('Dry run! No boundary will not be converted to spectra.')



    def spectra_to_waveseries(self, dry_run: bool=False, write_cache=False,
                                freq: tuple=(0, 10_000), **kwargs):
        self._dry_run = dry_run
        if self.spectra() is None:
            msg.warning('No Spectra to convert to WaveSeries!')
            return

        waveseries_reader = SpectraToWaveSeries(self.spectra(), freq)
        msg.header(waveseries_reader, 'Converting the spectra to wave series data...')
        name = self.spectra().name
        if not self.dry_run():
            self.import_waveseries(waveseries_reader=waveseries_reader,
                                    point_picker=bnd.pick.TrivialPicker(),
                                    name=name,
                                    write_cache=write_cache, **kwargs)
        else:
            msg.info('Dry run! No boundary will not be converted to spectra.')

    def boundary_to_waveseries(self, dry_run: bool=False, write_cache=False,
                                freq: tuple=(0, 10_000), **kwargs):
        self.boundary_to_spectra(dry_run=dry_run, write_cache=write_cache, **kwargs)
        self.spectra_to_waveseries(dry_run=dry_run, write_cache=write_cache, freq=freq, **kwargs)


    def set_spectral_grid_from_boundary(self):
        if self.boundary() is None:
            msg.warning("No Boundary exists. Can't set spectral grid.")
            return
        self.set_spectral_grid(freq=self.boundary().freq(), dirs=self.boundary().dirs())

    def set_spectral_grid_from_spectra(self, **kwargs):
        if self.spectra() is None:
            msg.warning("No Spectra exists. Can't set spectral grid.")
            return
        self.set_spectral_grid(freq=self.spectra().freq(), **kwargs)

    def set_spectral_grid(self, freq: np.ndarray=None, dirs: np.ndarray=None, freq0: float=0.04118, nfreq: int=32, ndir: int=36, finc: float=1.1, dirshift: float=0.):
        """Sets spectral grid for model run. Will be used to write input files."""
        if freq is None:
            freq = np.array([freq0*finc**n for n in np.linspace(0,nfreq-1,nfreq)])
        if dirs is None:
            dirs = np.linspace(0,360,ndir+1)[0:-1]+dirshift
        self._spectral_grid = SpectralGrid(name='spectral_grid', freq=freq, dirs=dirs)

    def export_grid(self, grid_writer: GridWriter=None,
                    filename: str=None, folder: str=None, dateformat: str=None,
                    format: str=None, dry_run=False) -> None:
        """Writes the grid data in the Grid-object to an external source,
        e.g. a file."""
        self._dry_run = dry_run
        if len(self.grid().topo())==0:
            msg.warning('Grid not meshed so nothing to export!')
            return

        # Try to use defaul grid writer if not provided
        self._grid_writer = grid_writer or self._get_grid_writer()
        if self._grid_writer is None:
            raise Exception('Define a GridWriter!')

        msg.header(self._grid_writer, f"Writing grid topography from {self.grid().name}")

        output_files = self._export_object('Grid',filename, folder, dateformat,
                            writer_function=self._grid_writer, format=format)

    def export_boundary(self, boundary_writer: BoundaryWriter=None,
                        filename: str=None, folder: str=None,
                        dateformat: str=None, format: str=None, dry_run=False) -> None:
        """Writes the spectra in the Boundary-object to an external source, e.g.
        a file."""
        self._dry_run = dry_run
        if self.boundary() is None and not self.dry_run(dry_run):
            raise Exception('Import boundary before exporting!')

        self._boundary_writer = boundary_writer or self._get_boundary_writer()
        if self._boundary_writer is None:
            raise Exception('Define a BoundaryWriter!')

        if self.boundary() is None:
            msg.header(self._boundary_writer, f"Writing boundary spectra from DryRunBoundary")
        else:
            msg.header(self._boundary_writer, f"Writing boundary spectra from {self.boundary().name}")

        if not self.dry_run():
            self.boundary()._set_convention(self._boundary_writer.convention())

        __ = self._export_object('Boundary', filename, folder, dateformat,
                            writer_function=self._boundary_writer, format=format)

    def export_spectra(self, spectral_writer: SpectralWriter=None,
                        filename: str=None, folder: str=None,
                        dateformat: str=None, format: str=None, dry_run=False) -> None:
        """Writes the spectra in the Spectra-object to an external source, e.g.
        a file."""
        self._dry_run = dry_run
        if self.spectra() is None and not self.dry_run():
            raise Exception('Import spectra before exporting!')

        self._spectral_writer = spectral_writer or self._get_spectral_writer()
        if self._spectral_writer is None:
            raise Exception('Define a SpectralWriter!')

        if self.spectra() is None:
            msg.header(self._spectral_writer, f"Writing omnidirectional spectra from DryRunSpectra")
        else:
            msg.header(self._spectral_writer, f"Writing omnidirectional spectra from {self.spectra().name}")

        if not self.dry_run():
            self.spectra()._set_convention(self._spectral_writer.convention())

        # Replace #Spectra etc and add file extension
        __ = self._export_object('Spectra', filename, folder, dateformat,
                            writer_function=self._spectral_writer, format=format)

    def export_waveseries(self, waveseries_writer: WaveSeriesWriter=None,
                        filename: str=None, folder: str=None,
                        dateformat: str=None, format: str=None, dry_run=False) -> None:
        """Writes the data of the WaveSeries-object to an external source, e.g.
        a file."""
        self._dry_run = dry_run
        if self.waveseries() is None and not self.dry_run():
            raise Exception('Import waveseries data before exporting!')

        self._waveseries_writer = waveseries_writer or self._get_waveseries_writer()
        if self._waveseries_writer is None:
            raise Exception('Define a WaveSeriesWriter!')

        if self.waveseries() is None:
            msg.header(self._waveseries_writer, f"Writing wave series data from DryRunSpectra")
        else:
            msg.header(self._waveseries_writer, f"Writing wave series data from {self.waveseries().name}")

        # Replace #Spectra etc and add file extension
        __ = self._export_object('WaveSeries', filename, folder, dateformat,
                            writer_function=self._waveseries_writer, format=format)

    def export_forcing(self, forcing_writer: ForcingWriter=None,
                        filename: str=None, folder: str=None,
                         dateformat: str=None, format: str=None, dry_run=False) -> None:
        """Writes the forcing data in the Forcing-object to an external source,
        e.g. a file."""
        self._dry_run = dry_run
        if self.forcing() is None and not self.dry_run():
            raise Exception('Import forcing before exporting!')

        self._forcing_writer = forcing_writer or self._get_forcing_writer()

        if self._forcing_writer is None:
            raise Exception('Define a ForcingWriter!')

        if self.forcing() is not None:
            msg.header(self._forcing_writer, f"Writing wind forcing from {self.forcing().name}")
        else:
            msg.header(self._forcing_writer, f"Writing wind forcing from DryRunForcing")


        __ = self._export_object('Forcing', filename, folder, dateformat,
                            writer_function=self._forcing_writer, format=format)

    def write_input_file(self, input_file_writer: InputFileWriter=None,
                        filename=None, folder=None, dateformat=None,
                        grid_path: str=None, forcing_path: str=None,
                        boundary_path: str=None, start_time: str=None,
                        end_time: str=None, dry_run=False) -> None:
        """Writes the grid data in the Grid-object to an external source,
        e.g. a file."""
        self._dry_run = dry_run
        self._input_file_writer = input_file_writer or self._get_input_file_writer()
        if self._input_file_writer is None:
            raise Exception('Define an InputFileWriter!')


        msg.header(self._input_file_writer, "Writing model input file...")

        # Controls generation of file names using the proper defaults etc.
        file_object = FileNames(dict_of_objects=self.dict_of_objects(),
                                filename=filename,
                                folder=folder,
                                dateformat=dateformat,
                                extension=self._input_file_writer._extension(),
                                obj_type='input_file',
                                edge_object='Grid')

        file_object.create_folder()


        if self.dry_run():
            msg.info('Dry run! No files will be written.')
            output_files = [file_object.get_filepath()]
        else:
            # Write the grid using the InputFileWriter object
            output_files = self._input_file_writer(dict_of_objects=self.dict_of_objects(),
                            filename=file_object.get_filepath())
            if type(output_files) is not list:
                output_files = [output_files]


        if self._input_file_writer._im_silent() or self.dry_run():
            for file in output_files:
                msg.to_file(file)

        return

    def run_model(self, model_executer: ModelExecuter=None,
                input_file: str=None, folder: str=None,
                dateformat: str=None, input_file_extension: str=None,
                dry_run: bool=False, mat_to_nc: bool=False) -> None:
        """Run the model."""
        self._dry_run = dry_run
        self._model_executer = model_executer or self._get_model_executer()
        if self._model_executer is None:
            raise Exception('Define a ModelExecuter!')

        # We always assume that the model is located in the folder the input
        # file was written to

        # Option 1) Use user provided
        # Option 2) Use knowledge of where has been exported
        # Option 3) Use default values to guess where is has previously been exported
        exported_path = Path(self.exported_to('input_file')[0])
        primary_file = input_file or exported_path.name
        primary_folder = folder #or str(exported_path.parent)

        if hasattr(self, '_input_file_writer'):
            extension = input_file_extension or self._input_file_writer._extension()
        else:
            extension = input_file_extension or 'swn'

        file_object = FileNames(dict_of_objects=self.dict_of_objects(),
                                filename=primary_file,
                                folder=primary_folder,
                                dateformat=dateformat,
                                extension=extension,
                                obj_type='model_executer',
                                edge_object='Grid')


        msg.header(self._model_executer, "Running model...")
        msg.plain(f"Using input file: {file_object.get_filepath()}")
        if not self.dry_run():
            self._model_executer(input_file=file_object.filename(), model_folder=file_object.folder())
        else:
            msg.info('Dry run! Model will not run.')
        if mat_to_nc:
            input_file = f'{file_object.folder()}/{self.grid().name}.mat'
            output_file = f'{file_object.folder()}/{self.grid().name}.nc'
            convert_swash_mat_to_netcdf(input_file=input_file,output_file=output_file, lon=self.grid().lon_edges(), lat=self.grid().lat_edges(), dt=1)

    def _export_object(self, obj_type, filename: str, folder: str, dateformat: str,
                    writer_function: WritingFunction, format: str) -> list[str]:

        # Controls generation of file names using the proper defaults etc.
        format = format or self._get_default_format()
        file_object = FileNames(format=format,
                                obj_type=obj_type,
                                dict_of_objects=self.dict_of_objects(),
                                filename=filename,
                                folder=folder,
                                dateformat=dateformat,
                                extension=writer_function._extension())
        if self.dry_run():
            msg.info('Dry run! No files will be written.')
            output_files = [file_object.get_filepath()]
        else:
            # Write the object using the WriterFunction
            file_object.create_folder()
            output_files = writer_function(self.dict_of_objects(), file_object)
            if type(output_files) is not list:
                output_files = [output_files]

        # Store name and location where file was written
        self.dict_of_objects().get(obj_type).exported_to = output_files
        # for file in output_files:
        #     self._exported_to[obj_type].append(file)

        if writer_function._im_silent() or self.dry_run():
            for file in output_files:
                msg.to_file(file)

        return output_files


    def dry_run(self):
        """Checks if method or global ModelRun dryrun is True.
        """
        return self._dry_run or self._global_dry_run


    def grid(self) -> str:
        """Returns the grid object."""
        return self._grid

    def forcing(self) -> Forcing:
        """Returns the forcing object if exists."""
        if hasattr(self, '_forcing'):
            return self._forcing
        else:
            return None

    def boundary(self) -> Boundary:
        """Returns the boundary object if exists."""
        if hasattr(self, '_boundary'):
            return self._boundary
        else:
            return None

    def spectra(self) -> Spectra:
        """Returns the spectral object if exists."""
        if hasattr(self, '_spectra'):
            return self._spectra
        else:
            return None

    def waveseries(self) -> WaveSeries:
        """Returns the wave series object if exists."""
        if hasattr(self, '_waveseries'):
            return self._waveseries
        else:
            return None

    def topo(self) -> Grid:
        """Returns the raw topography object if exists."""
        if hasattr(self.grid(), '_raw'):
            return self.grid().raw()
        else:
            return None
        
    def spectral_grid(self) -> Boundary:
        """Returns the spectral grid object if exists."""
        if hasattr(self, '_spectral_grid'):
            return self._spectral_grid
        else:
            return None

    def input_file(self) -> None:
        """Only defined to have method for all objects"""
        return None

    def dict_of_objects(self) -> dict[str: ModelRun, str: Grid, str: Forcing, str: Boundary, str: Spectra]:
        return {'ModelRun': self, 'Grid': self.grid(), 'Topo': self.topo(), 'Forcing': self.forcing(),
                'Boundary': self.boundary(), 'Spectra': self.spectra(), 'WaveSeries': self.waveseries(),
                'SpectralGrid': self.spectral_grid()}

    def list_of_objects(self) -> list[ModelRun, Grid, Forcing, Boundary, Spectra]:
        """[ModelRun, Boundary] etc."""
        return [x for x in list(self.dict_of_objects().values()) if x is not None]

    def list_of_object_strings(self) -> list[str]:
        """['ModelRun', 'Boundary'] etc."""
        return list(self.dict_of_objects().keys())

    def dict_of_object_names(self) -> dict[str: str]:
        """ {'Boundary': 'NORA3'} etc."""
        d = {}
        for a,b in self.dict_of_objects().items():
            if b is None:
                d[a] = None
            else:
                d[a] = b.name
        return d

    def exported_to(self, object: str) -> str:
        """Returns the path the object (e.g. grid) was exported to.

        If object has not been exported, the default filename is returned as
        a best guess
        """

        if eval(f'self.{object}()') is None:
            return ['']

        if self._exported_to.get(object) is not None:
            return self._exported_to[object]

        return ['']
    
    def time(self, crop: bool=False):
        """Returns times of ModelRun
        crop = True: Give the period that is covered by all objects (Forcing etc.)"""
        t0 = self._time[0]
        t1 = self._time[-1]

        if crop:
            for dnora_obj in self.list_of_objects():
                time = dnora_obj.time()
                if time[0] is not None:
                    t0 = pd.to_datetime([t0, time[0]]).max()
                if time[-1] is not None:
                    t1 = pd.to_datetime([t1, time[-1]]).min()
        time = pd.date_range(t0, t1, freq='H')
        return time[::len(time)-1]
    
    def start_time(self, crop: bool=False):
        """Returns start time of ModelRun
        crop = True: Give the period that is covered by all objects (Forcing etc.)"""
        return self.time(crop=crop)[0]

    def end_time(self, crop: bool=False):
        """Returns start time of ModelRun
        crop = True: Give the period that is covered by all objects (Forcing etc.)"""
        return self.time(crop=crop)[-1]

    def _get_default_format(self) -> str:
        return 'ModelRun'

    def _get_forcing_reader(self) -> ForcingReader:
        return None
    
    def _get_boundary_reader(self) -> BoundaryReader:
        return None
    
    def _get_point_picker(self) -> PointPicker:
        return None

    def _get_spectral_reader(self) -> SpectralReader:
        return None
    
    def _get_waveseries_reader(self) -> WaveSeriesReader:
        return None
    
    def _get_boundary_writer(self) -> BoundaryWriter:
        return bnd.write.DnoraNc()

    def _get_forcing_writer(self) -> ForcingWriter:
        return wnd.write.DnoraNc()

    def _get_spectral_writer(self) -> SpectralWriter:
        return spc.write.DnoraNc()

    def _get_waveseries_writer(self) -> WaveSeriesWriter:
        return wsr.write.DnoraNc()

    def _get_grid_writer(self) -> GridWriter:
        return grd.write.DnoraNc()