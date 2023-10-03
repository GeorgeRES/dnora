import numpy as np
from ...file_module import split_filepath, add_folder_to_filename


def ww3_grid(
    grid,
    filename: str,
    grid_exported_to: str,
    folder_on_server: str,
    freq1: float,
    nth: int,
    nk: int,
    dirshift: float,
) -> None:
    """Writes ww3_grid.nml file"""

    def write_block(fn: str):
        with open(f"{folder}{fn}", "r") as fin:
            block = fin.read()
        fout.write(block)

    def write_spectrum():
        thoff = dirshift * nth / 360
        fout.write("&SPECTRUM_NML\n")
        fout.write("  SPECTRUM%XFR         = 1.1\n")
        fout.write(f"  SPECTRUM%FREQ1       = {freq1:.5f}\n")
        fout.write(f"  SPECTRUM%NK          = {nk:.0f}\n")
        fout.write(f"  SPECTRUM%NTH         = {nth:.0f}\n")
        fout.write(f"  SPECTRUM%THOFF         = {thoff:.1f}\n")
        fout.write("/\n\n")

    def write_run():
        fout.write("&RUN_NML\n")
        fout.write("  RUN%FLCX       = T\n")
        fout.write("  RUN%FLCY       = T\n")
        fout.write("  RUN%FLTH       = T\n")
        fout.write("  RUN%FLCK       = T\n")
        fout.write("  RUN%FLSOU      = T\n")
        fout.write("/\n\n")

    def write_timestep():
        # Multiples of 10 s
        if grid.is_gridded():
            dtxy = np.floor(0.9 * grid.cfl(f0=freq1) / 10) * 10
        else:  # This needs to be corrected !!!!!!!!!!!!!!
            dtxy = 10.0

        if dtxy < 10:
            dtxy = 10
            dtkth = 10
        else:
            dtkth = dtxy / 2

        dtmax = 3 * dtxy

        if dtxy < 20:
            dtmin = 5
        else:
            dtmin = 10

        fout.write("&TIMESTEPS_NML\n")
        fout.write(f"  TIMESTEPS%DTMAX        =  {dtmax:.0f}.\n")
        fout.write(f"  TIMESTEPS%DTXY         =  {dtxy:.0f}.\n")
        fout.write(f"  TIMESTEPS%DTKTH        =  {dtkth:.0f}.\n")
        fout.write(f"  TIMESTEPS%DTMIN        =  {dtmin:.0f}.\n")

        fout.write("/\n\n")

    def write_grid():
        fout.write("&GRID_NML\n")
        fout.write(f"  GRID%NAME             = '{grid.name}'\n")
        fout.write("  GRID%NML              = 'namelists.nml'\n")
        if grid.is_gridded():
            fout.write("  GRID%TYPE             = 'RECT'\n")
        else:
            fout.write("  GRID%TYPE             = 'UNST'\n")
        if grid.is_cartesian():
            fout.write("  GRID%TYPE             = 'CART'\n")
        else:
            fout.write("  GRID%COORD            = 'SPHE'\n")
        fout.write("  GRID%CLOS             = 'NONE'\n")
        fout.write("  GRID%DMIN             = '2.0'\n")
        fout.write("/\n\n")

    def write_rect():
        sf = 1.0e06
        fout.write("&RECT_NML\n")
        fout.write(f"  RECT%NX               = {grid.nx():.0f}\n")
        fout.write(f"  RECT%NY               = {grid.ny():.0f}\n")
        fout.write(f"  RECT%SX               = {grid.dlon()*sf:.0f}.\n")
        fout.write(f"  RECT%SY               = {grid.dlat()*sf:.0f}.\n")
        fout.write(f"  RECT%SF               = {sf:.0f}.\n")
        fout.write(f"  RECT%X0               = {min(grid.lon()):.6f}\n")
        fout.write(f"  RECT%Y0               = {min(grid.lat()):.6f}\n")
        fout.write(f"  RECT%SF0              = 1.\n")
        fout.write("/\n\n")

    def write_cart():
        sf = 1.0e00
        fout.write("&RECT_NML\n")
        fout.write(f"  RECT%NX               = {grid.nx():.0f}\n")
        fout.write(f"  RECT%NY               = {grid.ny():.0f}\n")
        fout.write(f"  RECT%SX               = {grid.dx()*sf:.0f}.\n")
        fout.write(f"  RECT%SY               = {grid.dy()*sf:.0f}.\n")
        fout.write(f"  RECT%SF               = {sf:.0f}.\n")
        fout.write(f"  RECT%X0               = {min(grid.x()):.6f}\n")
        fout.write(f"  RECT%Y0               = {min(grid.y()):.6f}\n")
        fout.write(f"  RECT%SF0              = 1.\n")
        fout.write("/\n\n")

    def write_depth():
        fout.write("&DEPTH_NML\n")
        fout.write(f"  DEPTH%SF               = -1.\n")
        fout.write(
            f"  DEPTH%FILENAME         = {add_folder_to_filename(grid_exported_to[0],folder_on_server)}\n"
        )
        fout.write(f"  DEPTH%IDLA             =  1\n")
        fout.write(f"  DEPTH%IDFM             = 2\n")
        fout.write(f"  DEPTH%FORMAT           = '(F15.6)\n")
        fout.write("/\n\n")

    def write_mask():
        fout.write("&MASK_NML\n")
        fout.write(
            f"  MASK%FILENAME         =  {add_folder_to_filename(grid_exported_to[-1],folder_on_server)}\n"
        )
        fout.write(f"  MASK%IDLA             =  1\n")
        fout.write(f"  MASK%IDFM             = 2\n")
        fout.write(f"  MASK%FORMAT           = '(I3)\n")
        fout.write("/\n\n")

    def write_unst():
        fout.write("&UNST_NML\n")
        fout.write(f"  UNST%SF               = -1.\n")
        fout.write(
            f"  UNST%FILENAME         =  {add_folder_to_filename(grid_exported_to[0],folder_on_server)}\n"
        )
        fout.write(f"  UNST%IDLA             =  4\n")
        fout.write("/\n\n")

    def write_inbnd():
        blocks = []
        connect_flags = []
        ct = 0
        in_segment = False
        for n, point in enumerate(grid.boundary_mask()):
            if point:
                if not in_segment:
                    blocks.append(n + 1)
                    connect_flags.append("F")
                    in_segment = True
                    ct += 1
            else:
                if in_segment:
                    in_segment = False
                    if n > (blocks[-1] + 1):  # Block longer than one needs to be ended
                        blocks.append(n)
                        connect_flags.append("T")
                        ct += 1

        fout.write("&INBND_COUNT_NML\n")
        fout.write(f"  INBND_COUNT%N_POINT    = {ct:.0f}\n")
        fout.write("/\n\n")
        fout.write("&INBND_POINT_NML\n")
        for n, (block, flag) in enumerate(zip(blocks, connect_flags)):
            fout.write(f"  INBND_POINT({n+1:.0f})         = {block:.0f} 1 {flag}\n")
        fout.write("/\n\n")

    folder = __file__[:-17] + "/metadata/ww3_grid/"
    for n in range(len(grid_exported_to)):
        fn, __ = split_filepath(grid_exported_to[n])
        grid_exported_to[n] = fn
    with open(filename, "w") as fout:
        write_block("header.txt")
        fout.write("\n")
        write_block("spectrum.txt")
        write_spectrum()
        write_block("run.txt")
        write_run()
        write_block("timesteps.txt")
        write_timestep()
        write_block("grid.txt")
        write_grid()
        if grid.is_gridded():
            if grid.is_cartesian():
                write_block("cart.txt")
                write_cart()
            else:
                write_block("rect.txt")
                write_rect()
            write_block("depth.txt")
            write_depth()
            write_block("mask.txt")
            write_mask()

        else:
            write_block("unst.txt")
            write_unst()
            write_block("inbnd.txt")
            write_inbnd()
        write_block("footer.txt")


def ww3_prnc(
    filename: str,
    forcing_exported_to: list[str],
    folder_on_server: str,
    minwind: float = None,
) -> None:
    """Writes ww3_prnc.nml file"""

    def write_block(fn: str):
        with open(f"{folder}{fn}", "r") as fin:
            block = fin.read()
        fout.write(block)

    def write_forcing():
        fout.write("&FORCING_NML\n")
        fout.write("FORCING%FIELD%WINDS          = T\n")
        fout.write("FORCING%GRID%LATLON          = T\n")
        if minwind is not None:
            fout.write(f"FORCING%MINWIND              = {minwind}\n")
        fout.write("/\n")

    def write_file():
        fout.write("&FILE_NML\n")
        fout.write(
            f"  FILE%FILENAME      = '{add_folder_to_filename(forcing_exported_to[-1],folder_on_server)}'\n"
        )
        fout.write("  FILE%LONGITUDE     = 'lon'\n")
        fout.write("  FILE%LATITUDE      = 'lat'\n")
        fout.write("  FILE%VAR(1)        = 'u'\n")
        fout.write("  FILE%VAR(2)        = 'v'\n")
        fout.write("/\n")

    folder = __file__[:-17] + "/metadata/ww3_prnc/"
    for n in range(len(forcing_exported_to)):
        fn, __ = split_filepath(forcing_exported_to[n])
        forcing_exported_to[n] = fn
    with open(filename, "w") as fout:
        write_block("header.txt")
        fout.write("\n")
        write_block("forcing.txt")
        write_forcing()
        write_block("file.txt")
        write_file()
        write_block("footer.txt")


def ww3_specfile_list(outfile: str, list_of_filenames: list[str]):
    for n in range(len(list_of_filenames)):
        fn, __ = split_filepath(list_of_filenames[n])
        list_of_filenames[n] = fn
    with open(outfile, "w") as fout:
        for fn in list_of_filenames:
            fout.write(f"{fn}\n")


def ww3_bounc(filename: str, method: int, verbose_level: int):
    def write_block(fn: str):
        with open(f"{folder}{fn}", "r") as fin:
            block = fin.read()
        fout.write(block)

    def write_bound():
        fout.write("&BOUND_NML\n")
        fout.write("  BOUND%MODE = 'WRITE'\n")
        fout.write(f"  BOUND%INTERP = {method}\n")
        fout.write(f"  BOUND%VERBOSE = {verbose_level}\n")
        fout.write("  BOUND%FILE = 'spectral_boundary_files.list'\n")
        fout.write("/\n")

    folder = __file__[:-17] + "/metadata/ww3_bounc/"

    with open(filename, "w") as fout:
        write_block("header.txt")
        write_block("bound.txt")
        write_bound()
        write_block("footer.txt")


def ww3_spectral_output_list(
    filename: str, lons: np.ndarray, lats: np.ndarray, names: list[str] = None
):
    if names is None:
        names = [f"out_point{n+1:04.0f}" for n in range(len(lons))]
    with open(filename, "w") as fout:
        for lon, lat, name in zip(lons, lats, names):
            fout.write(f"{lon:11.8f} {lat:11.8f} '{name}'\n")


def ww3_shel(
    filename: str,
    folder_on_server: str,
    start_time: str,
    end_time: str,
    forcing: dict[str, bool],
    homog: dict[str, tuple[float, float]],
):
    def write_block(fn: str):
        with open(f"{folder}{fn}", "r") as fin:
            block = fin.read()
        fout.write(block)

    def write_domain():
        fout.write("&DOMAIN_NML\n")
        fout.write("  DOMAIN%IOSTYP  = 1\n")
        fout.write(f"  DOMAIN%START  = {start_time}\n")
        fout.write(f"  DOMAIN%STOP  = {end_time}\n")
        fout.write("/\n")

    def write_input():
        fout.write("&INPUT_NML\n")
        FORCING_NAMES = {"wnd": "WINDS", "wlv": "WATER_LEVELS", "ocr": "CURRENTS"}
        for field_type in FORCING_NAMES:
            if homog.get(field_type) is not None:
                fout.write(
                    f"  INPUT%FORCING%{FORCING_NAMES[field_type]}         = 'H'\n"
                )
            elif forcing.get(field_type):
                fout.write(
                    f"  INPUT%FORCING%{FORCING_NAMES[field_type]}         = 'T'\n"
                )
            else:
                fout.write(
                    f"  INPUT%FORCING%{FORCING_NAMES[field_type]}         = 'F'\n"
                )
        fout.write("/\n")

    def write_output():
        fout.write("&OUTPUT_TYPE_NML\n")
        fout.write(
            "  TYPE%FIELD%LIST     = 'HS LM TP DIR SPR DP T02 T0M1 T01 UST CHA DPT WND USS TUS TAW TWO TOC FAW FOC PHS PTP PTM10 PT01 PT02 PDIR PDP MXE MXH MXHC SDMH SDMHC ABR UBR FBB TBB CGE WCC WBT'\n"
        )
        fout.write(
            f"  TYPE%POINT%FILE     = '{add_folder_to_filename('spectral_points.list',folder_on_server)}'\n"
        )
        fout.write("/\n")

    def write_date():
        fout.write("&OUTPUT_DATE_NML\n")
        start_times = {"FIELD": start_time, "POINT": start_time, "RESTART": end_time}
        end_times = {"FIELD": end_time, "POINT": end_time, "RESTART": end_time}
        dt = {"FIELD": "3600", "POINT": "3600", "RESTART": "3600"}
        for output_type in ["FIELD", "POINT", "RESTART"]:
            fout.write(
                f"  DATE%{output_type}%START         = '{start_times[output_type]}'\n"
            )
            fout.write(
                f"  DATE%{output_type}%STOP          = '{end_times[output_type]}'\n"
            )
            fout.write(f"  DATE%{output_type}%STRIDE        = '{dt[output_type]}'\n")

        fout.write("/\n")

    def write_homog():
        FORCING_NAMES = {"wnd": "WND", "wlv": "LEV", "ocr": "CUR"}
        fout.write("&HOMOG_COUNT_NML\n")
        for field_type in FORCING_NAMES:
            if homog.get(field_type) is not None:
                fout.write(f"  HOMOG_COUNT%N_{FORCING_NAMES[field_type]}    = 1\n")
        fout.write("/\n")
        fout.write("&HOMOG_INPUT_NML\n")
        for field_type in FORCING_NAMES:
            data = homog.get(field_type)
            if data is not None:
                if isinstance(data, int) or isinstance(data, float):
                    data = [data]

                fout.write(
                    f"  HOMOG_INPUT(1)%NAME       = '{FORCING_NAMES[field_type]}'\n"
                )
                fout.write(f"  HOMOG_INPUT(1)%DATE   = {start_time}\n")

                for n, val in enumerate(data):
                    fout.write(f"  HOMOG_INPUT(1)%VALUE{n+1}     = {val}\n")

        fout.write("/\n")

    folder = __file__[:-17] + "/metadata/ww3_shel/"
    with open(filename, "w") as fout:
        write_block("header.txt")
        write_block("domain.txt")
        write_domain()
        write_block("input.txt")
        write_input()
        write_block("output.txt")
        write_output()
        write_block("date.txt")
        write_date()
        write_block("homog.txt")
        write_homog()
        write_block("footer.txt")