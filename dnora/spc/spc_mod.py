from abc import ABC, abstractmethod
from copy import copy
import numpy as np
import xarray as xr
from typing import List
import pandas as pd
# Import objects
from ..grd.grd_mod import Grid
from .process import spectral_processor_for_convention_change
from ..bnd.conventions import SpectralConvention, convention_from_string, convert_2d_to_1d
# Import abstract classes and needed instances of them
from ..bnd.pick import PointPicker, TrivialPicker
from .read import SpectralReader
from .. import msg
from ..cacher import Cacher
from ..skeletons.point_skeleton import PointSkeleton
from ..skeletons.coordinate_factory import add_time, add_frequency
from ..skeletons.mask_factory import add_mask
from ..skeletons.datavar_factory import add_datavar

from .process import SpectralProcessor

@add_mask(name='bad', coords='all', default_value=0)
@add_datavar(name='spec', coords='all', default_value=0.)
@add_datavar(name='mdir', coords='all', default_value=0.)
@add_datavar(name='spr', coords='all', default_value=0.)
@add_frequency(grid_coord=False)
@add_time(grid_coord=True)
class Spectra(PointSkeleton):
    def __init__(self, grid: Grid, name: str="AnonymousSpectra"):
        self._grid = grid
        self._name = name
        self._convention = None
        self._history = []

    def import_spectra(self, start_time: str, end_time: str,
                        spectral_reader: SpectralReader,
                        point_picker: PointPicker,
                        expansion_factor: float=1.5,
                        write_cache: bool=False,
                        read_cache: bool=False,
                        cache_name: str=None) -> None:

        """Imports omnidirectional spectra from a certain source.

        Spectra are import between start_time and end_time from the source
        defined in the spectral_reader. Which spectra to choose spatially
        are determined by the point_picker.
        """
        # Prepare for working with cahced data if we have to
        if write_cache or read_cache:
            cacher = Cacher(self, spectral_reader.name(), cache_name)


        # Read whatever we have in the chached data to start with
        # Setting the reader to read standard DNORA netcdf-files
        if read_cache and not cacher.empty():
            msg.info('Reading spectral data from cache!!!')
            original_spectral_reader = copy(spectral_reader)
            spectral_reader = DnoraNc(files=glob.glob(f'{cacher.filepath(extension=False)}*'), convention = spectral_reader.convention())

        self._history.append(copy(spectral_reader))

        msg.header(spectral_reader, "Reading coordinates of spectra...")
        lon_all, lat_all = spectral_reader.get_coordinates(self.grid(), start_time)

        msg.header(point_picker, "Choosing spectra...")
        inds = point_picker(self.grid(), lon_all, lat_all, expansion_factor)

        msg.header(spectral_reader, "Loading omnidirectional spectra...")
        time, freq, spec, mdir, spr, lon, lat, x, y, attributes = spectral_reader(self.grid(), start_time, end_time, inds)

        self._init_structure(x, y, lon, lat, time=time, freq=freq)

        self.ds_manager.set(spec, 'spec', coord_type='all')
        self.ds_manager.set(mdir, 'mdir', coord_type='all')
        self.ds_manager.set(spr, 'spr', coord_type='all')

        self.ds_manager.set_attrs(attributes)

        # Patch data if read from cache and all data not found
        if read_cache and not cacher.empty():
            patch_start, patch_end = cacher.determine_patch_periods(start_time, end_time)
            if patch_start:
                msg.info('Not all data found in cache. Patching from original source...')

                for t0, t1 in zip(patch_start, patch_end):
                    spectra_temp = Boundary(self.grid())
                    spectra_temp.import_spectra(self.grid(), start_time=t0, end_time=t1,
                                    spectral_reader=original_spectral_reader,
                                    point_picker=point_picker)
                    self._absorb_object(spectra_temp, 'time')

        # Dump monthly netcdf-files that will now be in standard DNORA format
        if write_cache:
            msg.info('Caching data:')
            cacher.write_cache()

        # E.g. are the spectra oceanic convention etc.
        self._convention = spectral_reader.convention()

    def process_spectra(self, spectral_processors: List[SpectralProcessor]=None):
        """Process all the individual spectra of the spectra object.

        E.g. change convention form WW3 to Oceanic, interpolate spectra to
        new frequency grid, or multiply everything with a constant.
        """

        if spectral_processors is None:
            msg.info("No SpectralProcessor provided. Doing Nothing.")
            return

        if not isinstance(spectral_processors, list):
            spectral_processors = [spectral_processors]

        convention_warning = False

        for processor in spectral_processors:

            msg.process(f"Processing spectra with {type(processor).__name__}")
            self._history.append(copy(processor))
            old_convention = processor._convention_in()
            if old_convention is not None:
                if old_convention != self.convention():
                    msg.warning(f"Spectral convention ({self.convention()}) doesn't match that expected by the processor ({old_convention})!")
                    convention_warning=True


            new_spec, new_dirs, new_freq, new_spr = processor(self.spec(), self.mdir(), self.freq(), self.spr())
            self._init_structure(x=self.x(strict=True), y=self.y(strict=True),
                            lon=self.lon(strict=True), lat=self.lat(strict=True),
                            time=self.time(), freq=new_freq)
            self.ds_manager.set(new_spec, 'spec', coord_type='all')
            self.ds_manager.set(new_dirs, 'mdir', coord_type='all')
            self.ds_manager.set(new_spr, 'spr', coord_type='all')
            # self.data.spec.values = new_spec
            # self.data = self.data.assign_coords(dirs=new_dirs)
            # self.data = self.data.assign_coords(freq=new_freq)

            # Set new convention if the processor changed it
            new_convention = processor._convention_out()
            if new_convention is not None:
                self._convention = new_convention
                if convention_warning:
                    msg.warning(f"Convention variable set to {new_convention}, but this might be wrong...")
                else:
                    msg.info(f"Changing convention from {old_convention} >>> {new_convention}")

            print(processor)
            msg.blank()
        return

    def _set_convention(self, convention: SpectralConvention) -> None:
        spectral_processor = spectral_processor_for_convention_change(
                            current_convention = self.convention(),
                            wanted_convention = convert_2d_to_1d(convention))

        if spectral_processor is None:
            msg.info(f"Convention ({self.convention()}) already equals wanted convention ({convention}).")
        else:
            self.process_spectra(spectral_processor)

    def convention(self):
        """Returns the convention (OCEAN/MET/MATH) of the spectra"""
        if not hasattr(self, '_convention'):
            return None
        return copy(self._convention)

    def grid(self) -> Grid:
        if hasattr(self, '_grid'):
            return self._grid
        return None

    def __str__(self) -> str:
        """Prints status of spectra."""

        msg.header(self, f"Status of spectra {self.name}")
        if self.x() is not None:
            msg.plain(f"Contains data ({len(self.x())} points) for {self.start_time()} - {self.end_time()}")
            msg.plain(f"Data covers: lon: {min(self.lon())} - {max(self.lon())}, lat: {min(self.lat())} - {max(self.lat())}")
        if len(self._history) > 0:
            msg.blank()
            msg.plain("Object has the following history:")
            for obj in self._history:
                msg.process(f"{obj.__class__.__bases__[0].__name__}: {type(obj).__name__}")
        #msg.print_line()
        #msg.plain("The Boundary is for the following Grid:")
        #print(self.grid())

        msg.print_line()

        return ''
