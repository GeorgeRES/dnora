from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from copy import copy
from typing import TYPE_CHECKING, Tuple

# Import objects
if TYPE_CHECKING:
    from .wnd_mod import Forcing # Boundary object

# Import default values and aux_funcsiliry functions
from .. import msg

from nco import Nco
from ..aux_funcs import write_monthly_nc_files

class ForcingWriter(ABC):
    """Writes the forcing data to a certain file format.

    This object is provided to the .export_forcing() method.
    """
    @abstractmethod
    def _extension(self):
        pass

    def _im_silent(self) -> bool:
        """Return False if you want to be responsible for printing out the
        file names."""
        return True

    @abstractmethod
    def __call__(self, forcing: Forcing, file_object: str) -> List[str]:
        """Writed the data from the Forcing object and returns the file and
        folder where data were written."""

        return output_file

class Null(ForcingWriter):
    def _extension(self):
        return 'junk'

    def __call__(self, dict_of_objects: dict, file_object):
        return ''

class DnoraNc(ForcingWriter):
    def _extension(self) -> str:
        return 'nc'

    def __call__(self, dict_of_objects: dict, file_object) -> tuple[str, str]:
        output_files = write_monthly_nc_files(dict_of_objects['Forcing'], file_object)
        return output_files

class WW3(ForcingWriter):
    """Writes wind forcing data to WAVEWATH III netcdf format."""
    def _extension(self):
        return 'nc'

    def __call__(self, dict_of_objects: dict, file_object) -> List[str]:
        filename = file_object.get_filepath()
        forcing = dict_of_objects['Forcing']
        forcing.ds().to_netcdf(filename)
        # WW3 need time to be first
        #nco = Nco()
        #nco.ncpdq(input=filename, output=filename, options=['-a', f'time,{forcing.y_str},{forcing.y_str}'])
        return filename


class SWAN(ForcingWriter):
    """Writes wind forcing data to SWAN ascii format."""

    def _extension(self):
        return 'asc'

    def __call__(self, dict_of_objects: dict, file_object) -> List[str]:
        filename = file_object.get_filepath()
        forcing = dict_of_objects['Forcing']

        days = forcing.days()
        with open(filename, 'w') as file_out:
            ct = 0
            for day in days:
                msg.plain(day.strftime('%Y-%m-%d'))
                times = forcing.times_in_day(day)
                for n in range(len(times)):
                    time_stamp = pd.to_datetime(
                        times[n]).strftime('%Y%m%d.%H%M%S')+'\n'
                    file_out.write(time_stamp)
                    np.savetxt(file_out, forcing.u()
                               [ct, :, :]*1000, fmt='%i')
                    file_out.write(time_stamp)
                    np.savetxt(file_out, forcing.v()
                               [ct, :, :]*1000, fmt='%i')
                    ct += 1

        return filename
