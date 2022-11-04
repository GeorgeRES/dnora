from pathlib import Path
import yaml
from dataclasses import dataclass
import pandas as pd
import re
from . import msg

def add_prefix(filename: str, prefix: str) -> str:
    """Adds a prefix to a filename, e.g. FileName.txt -> new_FileName.txt"""
    if prefix == '':
        return filename

    if prefix[-1] == '_':
        prefix = prefix[:-1]

    if filename == '':
            return prefix

    if filename[0] == '_':
        filename = filename[1:]

    return f'{prefix}_{filename}'

def add_suffix(filename: str, suffix: str) -> str:
    """Adds a suffix to a filename, e.g. FileName.txt -> FileName_new.txt"""
    if suffix == '':
        return filename

    if suffix[0] == '_':
        suffix = suffix[1:]

    if filename == '':
        return suffix

    filename_list = filename.split('.')

    if len(filename_list) == 1:
        return f'{filename}_{suffix}'
    else:
        filename = '.'.join(filename_list[0:-1])
        extension = f'{filename_list[-1]}'
        return f'{filename}_{suffix}.{extension}'

def replace_times(filename: str, dateformat: str, times: list) -> str:
    """Substitutes the strings #T0, #T1, #T2... etc. in filname with time
    stamps from a list of times, using the format in dateformat.

    e.g. #T0_file.txt, ['2020-05-04 18:00'], %Y%m%d%H%M -> 202005041800_file.txt
    """

    for ct, t in enumerate(times):
        if t is not None:
            filename = re.sub(f"#T{ct}", pd.Timestamp(t).strftime(dateformat), filename)

    return filename

def replace_lonlat(filename: str, lon: float, lat: float) -> str:
    """Substitutes the strings #Lon, #Lat in filename with values of lon and
    lat.

    e.g. #Lon_#Lat_file.txt, 8.0, 60.05 -> 08.0000000_60.05000000_file.txt
    """
    if isinstance(lon, tuple):
        if lon[0] is not None:
            filename = re.sub("#Lon0", f"{lon[0]:010.7f}", filename)
        if lon[1] is not None:
            filename = re.sub("#Lon1", f"{lon[1]:010.7f}", filename)
    else:
        if lon is not None:
            filename = re.sub("#Lon", f"{lon:010.7f}", filename)

    if isinstance(lon, tuple):
        if lat[0] is not None:
            filename = re.sub("#Lat0", f"{lat[0]:010.7f}", filename)
        if lat[1] is not None:
            filename = re.sub("#Lat1", f"{lat[1]:010.7f}", filename)
    else:
        if lon is not None:
            filename = re.sub("#Lat", f"{lat:010.7f}", filename)

    return filename

def replace_xy(filename: str, x: float, y: float) -> str:
    """Substitutes the strings #X, #Y in filename with values of lon and
    lat.

    e.g. #Lon_#Lat_file.txt, 8.0, 60.05 -> 08.0000000_60.05000000_file.txt
    """
    if isinstance(x, tuple):
        if x[0] is not None:
            filename = re.sub("#X0", f"{x[0]:010.3f}", filename)
        if x[1] is not None:
            filename = re.sub("#Y1", f"{x[1]:010.3f}", filename)
    else:
        if x is not None:
            filename = re.sub("#X", f"{x:010.3f}", filename)

    if isinstance(y, tuple):
        if y[0] is not None:
            filename = re.sub("#Y0", f"{y[0]:010.3f}", filename)
        if y[1] is not None:
            filename = re.sub("#Y1", f"{y[1]:010.3f}", filename)
    else:
        if y is not None:
            filename = re.sub("#Y", f"{y:010.3f}", filename)

    return filename

def replace_objects(filename: str, dict_of_object_names: dict[str: str]) -> str:
    """Substitutes the strings #{Object} in filename with the name given to
    the object.

    e.g. #Grid_#Forcing_file.txt, [Grid(..., name="Sula"), Forcing(..., name='NORA3')]
        -> Sula_NORA3_file.txt
    """

    for obj_type, obj_name in dict_of_object_names.items():
        if obj_name is not None:
            filename = re.sub(f"#{obj_type}", obj_name, filename)
            filename = re.sub(f"#{obj_type.title()}", obj_name, filename)

    return filename

def get_list_of_placeholders():
    defaults_file = Path(__file__).parent.joinpath(Path('defaults.yml'))
    with open(defaults_file, 'r') as file:
      defaults = yaml.safe_load(file)
    return defaults['list_of_placeholders']

def clean(filename: str, list_of_placeholders: list[str]=None) -> str:
    """ Cleans out the file name from possible used placeholders, e.g. #Grid
    as given in the list.

    Also removes multiple underscores '___' etc.
    """
    if list_of_placeholders is None:
        list_of_placeholders = get_list_of_placeholders()

    for s in list_of_placeholders:
            filename = re.sub(s, '', filename)

    filename = re.sub("_{2,10}", '_', filename)
    filename = re.sub("_-_", '', filename)
    filename = re.sub("_-", '_', filename)
    if filename and filename[-1] == '_':
        filename = filename[:-1]

    return filename

def get_default_value(key: str, obj_type: str, primary: dict, fallback: dict):
    """Get a key (e.g. folder) from the defaults list.

    1) Tries Model+dnora_obj specific value (e.g. SWAN-wnd-folder)

    2) Tries Model specifc values (e.g. SWAN-folder)

    3) Returns ModelRun defaults (e.g. ModulRun-wnd-folder)
    """

    # Try dnora_obj specific fallback name
    fallback_name = None
    if fallback.get(obj_type) is not None:
        fallback_name = fallback[obj_type].get(key)

    # Try object non-specific fallback name
    fallback_name = fallback_name or fallback.get(key)

    # Try dnora_obj specific primary name
    primary_name = None
    if primary.get(obj_type) is not None:
        primary_name = primary[obj_type].get(key)

    # Try dnora_obj non-specific primary name
    primary_name = primary_name or primary.get(key)

    final_name = primary_name or fallback_name

    if final_name is None:
        raise ValueError("Could not find any default name!")

    return final_name

def add_folder_to_filename(filename: str, folder: str) -> str:
    return str(Path(folder).joinpath(filename))

def split_filepath(filepath: str) -> tuple[str, str]:
    folder = str(Path(filepath).parent)
    filename = Path(filepath).name
    return filename, folder

@dataclass
class FileNames:
    format: str
    dnora_obj: str
    dict_of_object_names: list=None
    clean_names: bool=True
    _filename: str=None
    _folder: str=None
    _dateformat: str=None
    edges_from_grid: bool=False
    extension: str=None
    start_time: str = None
    end_time: str = None

    def __post_init__(self):
        defaults_file = Path(__file__).parent.joinpath(Path('defaults.yml'))
        with open(defaults_file, 'r') as file:
          self._defaults = yaml.safe_load(file)
        self.fallback = self._defaults['ModelRun']
        self.primary = self._defaults[self.format]
        if self.dict_of_object_names is None:
            self.dict_of_object_names = {}

        dict_keys = [key.lower() for key in self.dict_of_object_names.keys()]
        if isinstance(self.dnora_obj, str):
            self.obj_type = self.dnora_obj
        else:
            self.obj_type = type(self.dnora_obj).__name__.lower()
            if not self.obj_type in dict_keys:
                self.dict_of_object_names[self.obj_type] = self.dnora_obj.name
        if self.start_time is not None:
            self.start_time = pd.to_datetime(self.start_time)
        if self.end_time is not None:
            self.end_time = pd.to_datetime(self.end_time)
        #self.placeholders = self._defaults['list_of_placeholders']

    def dateformat(self) -> str:
        return self._dateformat or get_default_value('dateformat', self.obj_type, self.primary, self.fallback)

    def filename(self, extension: str=None, start_time: str=None, end_time: str=None, plot: bool=False) -> str:
        if plot:
            key = 'plotname'
        else:
            key = 'filename'
        filename = self._filename or get_default_value(key, self.obj_type, self.primary, self.fallback)
        filename = self.replace_placeholders(filename, start_time, end_time)
        extension = extension or self.extension
        if extension is None:
            return Path(filename)
        return f'{Path(filename)}.{extension}'

    def folder(self, plot: bool=False) -> str:
        if plot:
            key = 'plotfolder'
        else:
            key = 'folder'
        folder = self._folder or get_default_value(key, self.obj_type, self.primary, self.fallback)

        return Path(self.replace_placeholders(folder))


    def filepath(self, extension: str=None, start_time: str=None, end_time: str=None) -> str:
        return add_folder_to_filename(self.filename(extension, start_time, end_time), self.folder())

    def create_folder(self, plot: bool=False) -> None:
        folder = Path(self.folder(plot=plot))

        if not folder.is_dir():
            msg.plain(f"Creating folder {str(folder)}")
            folder.mkdir(parents=True)

    def replace_placeholders(self, unclean_string: str, start_time: str=None, end_time: str=None) -> str:
        unclean_string = replace_objects(unclean_string, self.dict_of_object_names)
        if not isinstance(self.dnora_obj, str):
            if self.edges_from_grid:
                lon = self.dnora_obj.grid().edges('lon', strict=True)
                lat = self.dnora_obj.grid().edges('lat', strict=True)
                x = self.dnora_obj.grid().edges('x', strict=True)
                y = self.dnora_obj.grid().edges('y', strict=True)
            else:
                lon = self.dnora_obj.edges('lon', strict=True)
                lat = self.dnora_obj.edges('lat', strict=True)
                x = self.dnora_obj.edges('x', strict=True)
                y = self.dnora_obj.edges('y', strict=True)

            unclean_string = replace_lonlat(unclean_string, lon, lat)
            unclean_string = replace_xy(unclean_string, x, y)
            start_time = start_time or self.dnora_obj.time()[0]
            end_time = end_time or self.dnora_obj.time()[1]
            clean_string = replace_times(unclean_string, self.dateformat(), [start_time, end_time])
        else:
            clean_string = unclean_string

        if self.clean_names:
            clean_string = clean(clean_string)

        return clean_string
