from ..mdl import ModelRun
from .. import msg
from ..file_module import FileNames

from typing import Union
from ..grd.grd_mod import Grid, TriGrid
from . import grd, wnd, bnd, spc, wsr, wlv, inp
from .grd.grid_writers import GridWriter
from .wnd.forcing_writers import ForcingWriter
from .bnd.boundary_writers import BoundaryWriter
from .spc.spectral_writers import SpectralWriter
from .wsr.waveseries_writers import WaveSeriesWriter
from .wlv.waterlevel_writers import WaterLevelWriter
from .inp.input_file_writers import InputFileWriter

WritingFunction = Union[
    GridWriter,
    ForcingWriter,
    BoundaryWriter,
    SpectralWriter,
    WaveSeriesWriter,
    WaterLevelWriter,
    InputFileWriter,
]


class DataExporter:
    def __init__(self, model: ModelRun):
        self.model = model

    def export_grid(
        self,
        grid_writer: GridWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the grid data in the Grid-object to an external source,
        e.g. a file."""
        if type(self.model.grid()) is Grid:
            writer_function = self._setup_export("Grid", grid_writer, dry_run)
        elif type(self.model.grid()) is TriGrid:
            writer_function = self._setup_export("TriGrid", grid_writer, dry_run)

        self._export_object(
            "Grid",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def export_boundary(
        self,
        boundary_writer: BoundaryWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the spectra in the Boundary-object to an external source, e.g.
        a file."""
        writer_function = self._setup_export("Boundary", boundary_writer, dry_run)

        if not self.dry_run():
            self.model.boundary()._set_convention(writer_function.convention())

        self._export_object(
            "Boundary",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def export_spectra(
        self,
        spectral_writer: SpectralWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the spectra in the Spectra-object to an external source, e.g.
        a file."""
        writer_function = self._setup_export("Spectra", spectral_writer, dry_run)

        if not self.dry_run():
            self.model.spectra()._set_convention(writer_function.convention())

        self._export_object(
            "Spectra",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def export_waveseries(
        self,
        waveseries_writer: WaveSeriesWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the data of the WaveSeries-object to an external source, e.g.
        a file."""
        writer_function = self._setup_export("WaveSeries", waveseries_writer, dry_run)

        self._export_object(
            "WaveSeries",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def export_forcing(
        self,
        forcing_writer: ForcingWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the forcing data in the Forcing-object to an external source,
        e.g. a file."""
        writer_function = self._setup_export("Forcing", forcing_writer, dry_run)

        self._export_object(
            "Forcing",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def export_waterlevel(
        self,
        waterlevel_writer: WaterLevelWriter = None,
        filename: str = None,
        folder: str = None,
        dateformat: str = None,
        format: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the forcing data in the Forcing-object to an external source,
        e.g. a file."""
        writer_function = self._setup_export("WaterLevel", waterlevel_writer, dry_run)

        self._export_object(
            "WaterLevel",
            filename,
            folder,
            dateformat,
            writer_function=writer_function,
            format=format,
            **kwargs,
        )

    def _export_object(
        self,
        obj_type,
        filename: str,
        folder: str,
        dateformat: str,
        writer_function: WritingFunction,
        format: str,
        **kwargs,
    ) -> list[str]:
        # Controls generation of file names using the proper defaults etc.
        if writer_function is None:
            msg.info(f"No {obj_type} data exists. Won't export anything.")
            return

        msg.header(
            writer_function,
            f"Writing {obj_type} data from {self.model[obj_type].name}",
        )

        format = format or self._get_default_format()
        file_object = FileNames(
            format=format,
            obj_type=obj_type,
            model=self.model,
            filename=filename,
            folder=folder,
            dateformat=dateformat,
        )
        if self.dry_run():
            msg.info("Dry run! No files will be written.")
            output_files = [file_object.get_filepath()]
        else:
            # Write the object using the WriterFunction
            file_object.create_folder()
            output_files = writer_function(self.model, file_object, **kwargs)
            if type(output_files) is not list:
                output_files = [output_files]

        # Store name and location where file was written
        self.model._exported_to[obj_type.lower()] = output_files
        # for file in output_files:
        #     self._exported_to[obj_type].append(file)

        if writer_function._im_silent() or self.dry_run():
            for file in output_files:
                msg.to_file(file)

    def write_input_file(
        self,
        input_file_writer: InputFileWriter = None,
        filename=None,
        folder=None,
        dateformat=None,
        format: str = None,
        grid_path: str = None,
        forcing_path: str = None,
        boundary_path: str = None,
        start_time: str = None,
        end_time: str = None,
        dry_run=False,
        **kwargs,
    ) -> None:
        """Writes the grid data in the Grid-object to an external source,
        e.g. a file."""
        self._dry_run = dry_run
        self._input_file_writer = input_file_writer or self._get_input_file_writer()

        if self._input_file_writer is None:
            msg.info("No InputFileWriter defines. Won't do anything.")
            return

        msg.header(self._input_file_writer, "Writing model input file...")

        # Controls generation of file names using the proper defaults etc.
        format = format or self._get_default_format()
        file_object = FileNames(
            model=self.model,
            filename=filename,
            folder=folder,
            dateformat=dateformat,
            obj_type="input_file",
            edge_object="Grid",
        )

        file_object.create_folder()

        if self.dry_run():
            msg.info("Dry run! No files will be written.")
            output_files = [file_object.get_filepath()]
        else:
            # Write the grid using the InputFileWriter object
            output_files = self._input_file_writer(
                self.model, filename=file_object.get_filepath(), **kwargs
            )
            if type(output_files) is not list:
                output_files = [output_files]

        if self._input_file_writer._im_silent() or self.dry_run():
            for file in output_files:
                msg.to_file(file)

        return

    def _setup_export(self, obj_type: str, writer_function, dry_run: bool):
        self._dry_run = dry_run
        if obj_type == "TriGrid":
            check_type = "Grid"
        else:
            check_type = obj_type

        if self.model[check_type] is None:
            return None

        writer_function = writer_function or self[f"{obj_type}_writer"]

        if writer_function is None:
            raise Exception(f"Define a {check_type}Writer!")

        return writer_function

    def __getitem__(self, writer: str):
        """writer = 'grid_writer, forcing_writer etc."""
        return eval(f"self._get_{writer.lower()}()")

    def dry_run(self) -> bool:
        return self._dry_run or self.model.dry_run()

    def _get_default_format(self) -> str:
        return "ModelRun"

    def _get_boundary_writer(self) -> BoundaryWriter:
        return bnd.DnoraNc()

    def _get_forcing_writer(self) -> ForcingWriter:
        return wnd.DnoraNc()

    def _get_spectra_writer(self) -> SpectralWriter:
        return spc.DnoraNc()

    def _get_waveseries_writer(self) -> WaveSeriesWriter:
        return wsr.DnoraNc()

    def _get_waterlevel_writer(self) -> WaterLevelWriter:
        return wlv.DnoraNc()

    def _get_grid_writer(self) -> GridWriter:
        return grd.DnoraNc()

    def _get_trigrid_writer(self) -> GridWriter:
        return grd.DnoraNc()

    def _get_input_file_writer(self) -> InputFileWriter:
        return None
