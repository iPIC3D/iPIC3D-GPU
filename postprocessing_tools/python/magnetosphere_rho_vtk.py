#!/usr/bin/env python3
"""Plot the midpoint-y XZ density slices from iPIC3D legacy VTK files."""

"""
python magnetosphere_rho_vtk.py --output fields_png/rho.gif --input-dir ./fields_rho  --species 0  --start-step 0 --stop-step 5000 --cmap viridis --absolute-colors 

Common Matplotlib colormap options:
	Sequential: viridis, plasma, inferno, magma, cividis, turbo
	Diverging:  coolwarm, seismic, RdBu_r, bwr, PiYG, BrBG
	Density:    hot, afmhot, gist_heat, cubehelix, gray
"""


import argparse
import re
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


COMMON_CMAPS = (
	"viridis",
	"plasma",
	"inferno",
	"magma",
	"cividis",
	"turbo",
	"coolwarm",
	"seismic",
	"RdBu_r",
	"bwr",
	"PiYG",
	"BrBG",
	"hot",
	"afmhot",
	"gist_heat",
	"cubehelix",
	"gray",
)

VTK_BINARY_DTYPES = {
	"char": np.dtype("i1"),
	"unsigned_char": np.dtype("u1"),
	"short": np.dtype(">i2"),
	"unsigned_short": np.dtype(">u2"),
	"int": np.dtype(">i4"),
	"unsigned_int": np.dtype(">u4"),
	"long": np.dtype(">i8"),
	"unsigned_long": np.dtype(">u8"),
	"float": np.dtype(">f4"),
	"double": np.dtype(">f8"),
}

FONT_SIZES = {
	"title": 20,
	"axis_label": 18,
	"axis_ticks": 16,
	"colorbar_label": 16,
	"colorbar_ticks": 16,
}


class VTKMetadata(NamedTuple):
	dimensions: Tuple[int, int, int]
	origin: Tuple[float, float, float]
	spacing: Tuple[float, float, float]
	point_count: int
	scalar_name: str
	dtype: np.dtype
	data_offset: int


def _three_values(tokens: List[str], value_type: type) -> tuple:
	if len(tokens) != 4:
		raise ValueError(f"Expected three values in: {' '.join(tokens)}")
	return tuple(value_type(value) for value in tokens[1:])


def read_vtk_metadata(path: Path) -> VTKMetadata:
	"""Read metadata from a binary legacy VTK STRUCTURED_POINTS file."""
	dimensions = None
	origin = None
	spacing = None
	point_count = None
	scalar_name = None
	scalar_type = None

	with path.open("rb") as vtk_file:
		signature = vtk_file.readline().decode("ascii").strip()
		if not signature.startswith("# vtk DataFile Version"):
			raise ValueError(f"{path} is not a legacy VTK file")

		vtk_file.readline()  # Title
		encoding = vtk_file.readline().decode("ascii").strip().upper()
		if encoding != "BINARY":
			raise ValueError(f"{path} uses {encoding!r}; only BINARY VTK is supported")

		dataset = vtk_file.readline().decode("ascii").strip().upper()
		if dataset != "DATASET STRUCTURED_POINTS":
			raise ValueError(f"{path} uses unsupported {dataset!r}")

		while True:
			raw_line = vtk_file.readline()
			if not raw_line:
				raise ValueError(f"{path} has no scalar data payload")

			tokens = raw_line.decode("ascii").strip().split()
			if not tokens:
				continue

			keyword = tokens[0].upper()
			if keyword == "DIMENSIONS":
				dimensions = _three_values(tokens, int)
			elif keyword == "ORIGIN":
				origin = _three_values(tokens, float)
			elif keyword in {"SPACING", "ASPECT_RATIO"}:
				spacing = _three_values(tokens, float)
			elif keyword == "POINT_DATA":
				point_count = int(tokens[1])
			elif keyword == "SCALARS":
				if len(tokens) not in {3, 4}:
					raise ValueError(f"Malformed SCALARS line in {path}")
				if len(tokens) == 4 and int(tokens[3]) != 1:
					raise ValueError(f"Only one-component scalar data is supported: {path}")
				scalar_name = tokens[1]
				scalar_type = tokens[2].lower()
			elif keyword == "LOOKUP_TABLE":
				data_offset = vtk_file.tell()
				break

	if None in {dimensions, origin, spacing, point_count, scalar_name, scalar_type}:
		raise ValueError(f"Incomplete STRUCTURED_POINTS metadata in {path}")
	if scalar_type not in VTK_BINARY_DTYPES:
		raise ValueError(f"Unsupported VTK scalar type {scalar_type!r} in {path}")

	expected_points = int(np.prod(dimensions))
	if point_count != expected_points:
		raise ValueError(
			f"POINT_DATA is {point_count}, but DIMENSIONS require {expected_points}: {path}"
		)

	dtype = VTK_BINARY_DTYPES[scalar_type]
	expected_size = data_offset + point_count * dtype.itemsize
	if path.stat().st_size < expected_size:
		raise ValueError(f"Scalar payload is truncated in {path}")

	return VTKMetadata(
		dimensions=dimensions,
		origin=origin,
		spacing=spacing,
		point_count=point_count,
		scalar_name=scalar_name,
		dtype=dtype,
		data_offset=data_offset,
	)


def read_xz_slice(
	path: Path, y_index: Optional[int] = None
) -> Tuple[VTKMetadata, int, np.ndarray, np.ndarray, np.ndarray]:
	"""Extract one XZ plane, using the midpoint grid point by default."""
	metadata = read_vtk_metadata(path)
	nx, ny, nz = metadata.dimensions
	selected_y = ny // 2 if y_index is None else y_index
	if not 0 <= selected_y < ny:
		raise ValueError(f"y index {selected_y} is outside [0, {ny - 1}]")

	grid = np.memmap(
		path,
		dtype=metadata.dtype,
		mode="r",
		offset=metadata.data_offset,
		shape=(nz, ny, nx),
		order="C",
	)
	density = np.array(grid[:, selected_y, :], dtype=np.float32)
	del grid

	x = metadata.origin[0] + np.arange(nx) * metadata.spacing[0]
	z = metadata.origin[2] + np.arange(nz) * metadata.spacing[2]
	return metadata, selected_y, x, z, density


def timestep_from_path(path: Path) -> int:
	match = re.search(r"_(-?\d+)$", path.stem)
	if match is None:
		raise ValueError(f"Cannot read a timestep from {path.name}")
	return int(match.group(1))


def discover_files(
	input_dir: Path,
	species: int,
	start_step: Optional[int],
	stop_step: Optional[int],
	stride: int,
) -> List[Tuple[int, Path]]:
	search_dir = input_dir
	pattern = f"*rho{species}_*.vtk"
	matching_paths = list(search_dir.glob(pattern))
	if not matching_paths and (input_dir / "data").is_dir():
		search_dir = input_dir / "data"
		matching_paths = list(search_dir.glob(pattern))
	files = sorted(
		((timestep_from_path(path), path) for path in matching_paths),
		key=lambda item: item[0],
	)
	files = [
		item
		for item in files
		if (start_step is None or item[0] >= start_step)
		and (stop_step is None or item[0] <= stop_step)
	]
	return files[::stride]


def create_gif(
	densities: List[np.ndarray],
	timesteps: List[int],
	x: np.ndarray,
	z: np.ndarray,
	y_coordinate: float,
	scalar_name: str,
	output: Path,
	fps: float,
	vmin: float,
	vmax: float,
	cmap: str,
	absolute_colors: bool,
) -> None:
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		from matplotlib.animation import FuncAnimation, PillowWriter
	except ModuleNotFoundError as error:
		raise SystemExit(
			"GIF generation requires matplotlib and Pillow. Install them with:\n"
			"  python3 -m pip install matplotlib pillow"
		) from error

	output.parent.mkdir(parents=True, exist_ok=True)
	figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
	display_name = f"|{scalar_name}|" if absolute_colors else scalar_name
	image = axis.imshow(
		np.abs(densities[0]) if absolute_colors else densities[0],
		extent=(x[0], x[-1], z[0], z[-1]),
		origin="lower",
		aspect="equal",
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		interpolation="nearest",
	)
	axis.set_xlabel(r"x [$d_i$]", fontsize=FONT_SIZES["axis_label"])
	axis.set_ylabel(r"z [$d_i$]", fontsize=FONT_SIZES["axis_label"])
	axis.tick_params(axis="both", labelsize=FONT_SIZES["axis_ticks"])
	title = axis.set_title("", fontsize=FONT_SIZES["title"])
	colorbar = figure.colorbar(image, ax=axis)
	colorbar.set_label(display_name, fontsize=FONT_SIZES["colorbar_label"])
	colorbar.ax.tick_params(labelsize=FONT_SIZES["colorbar_ticks"])

	def update(frame_index: int):
		density = densities[frame_index]
		image.set_data(np.abs(density) if absolute_colors else density)
		title.set_text(
			f"{display_name} on XZ plane | y = {y_coordinate:g} | "
			f"step = {timesteps[frame_index]}"
		)
		return image, title

	frames_dir = output.parent / f"{output.stem}_frames"
	frames_dir.mkdir(parents=True, exist_ok=True)
	frame_digits = max(4, len(str(len(densities) - 1)))
	for frame_index, timestep in enumerate(timesteps):
		update(frame_index)
		frame_path = frames_dir / (
			f"frame_{frame_index:0{frame_digits}d}_step_{timestep}.png"
		)
		figure.savefig(frame_path, dpi=110)
	print(f"Wrote {len(densities)} PNG frames to {frames_dir}")

	animation = FuncAnimation(
		figure,
		update,
		frames=len(densities),
		interval=1000.0 / fps,
		blit=False,
	)
	animation.save(output, writer=PillowWriter(fps=fps), dpi=110)
	plt.close(figure)


def parse_arguments() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent
	parser = argparse.ArgumentParser(
		description="Extract midpoint-y XZ density planes and animate their evolution."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=script_dir / "r1" / "fields_rho",
		help="directory containing Dipole3D_rho*_STEP.vtk files",
	)
	parser.add_argument("--species", type=int, default=0, help="density species index")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="output GIF path (default: rhoSPECIES_xz.gif beside this script)",
	)
	parser.add_argument("--y-index", type=int, default=None, help="y grid index")
	parser.add_argument("--start-step", type=int, default=None)
	parser.add_argument("--stop-step", type=int, default=None)
	parser.add_argument(
		"--stride", type=int, default=1, help="keep every Nth available timestep"
	)
	parser.add_argument("--fps", type=float, default=8.0)
	parser.add_argument("--vmin", type=float, default=None)
	parser.add_argument("--vmax", type=float, default=None)
	parser.add_argument(
		"--cmap",
		default="viridis",
		help=f"Matplotlib colormap; common options: {', '.join(COMMON_CMAPS)}",
	)
	parser.add_argument(
		"--absolute-colors",
		action="store_true",
		help="color by absolute density while retaining signed printed values",
	)
	parser.add_argument(
		"--print-values",
		action="store_true",
		help="print every value in each extracted 2D density array",
	)
	arguments = parser.parse_args()
	if arguments.stride < 1:
		parser.error("--stride must be at least 1")
	if arguments.fps <= 0:
		parser.error("--fps must be positive")
	if arguments.absolute_colors and arguments.vmin is not None and arguments.vmin < 0:
		parser.error("--vmin cannot be negative with --absolute-colors")
	return arguments


def main() -> None:
	arguments = parse_arguments()
	input_dir = arguments.input_dir.expanduser().resolve()
	if not input_dir.is_dir():
		raise SystemExit(f"Input directory does not exist: {input_dir}")

	selected_files = discover_files(
		input_dir,
		arguments.species,
		arguments.start_step,
		arguments.stop_step,
		arguments.stride,
	)
	if not selected_files:
		raise SystemExit(f"No matching density VTK files found in {input_dir}")

	densities = []
	timesteps = []
	reference_metadata = None
	x = z = None
	selected_y = None
	data_min = np.inf
	data_max = -np.inf

	for timestep, path in selected_files:
		metadata, current_y, current_x, current_z, density = read_xz_slice(
			path, arguments.y_index
		)
		if reference_metadata is None:
			reference_metadata = metadata
			selected_y = current_y
			x, z = current_x, current_z
		elif (
			metadata.dimensions != reference_metadata.dimensions
			or metadata.origin != reference_metadata.origin
			or metadata.spacing != reference_metadata.spacing
			or metadata.scalar_name != reference_metadata.scalar_name
		):
			raise ValueError(f"Grid or scalar metadata changed in {path}")

		densities.append(density)
		timesteps.append(timestep)
		print(
			f"step {timestep:>8}: {metadata.scalar_name} "
			f"min={np.nanmin(density):.7g} max={np.nanmax(density):.7g} "
			f"mean={np.nanmean(density):.7g}"
		)
		if arguments.absolute_colors:
			data_min = 0.0
			data_max = max(data_max, float(np.nanmax(np.abs(density))))
		else:
			data_min = min(data_min, float(np.nanmin(density)))
			data_max = max(data_max, float(np.nanmax(density)))
		if arguments.print_values:
			print(np.array2string(density, threshold=np.inf, max_line_width=160))

	assert reference_metadata is not None and selected_y is not None
	assert x is not None and z is not None
	y_coordinate = (
		reference_metadata.origin[1]
		+ selected_y * reference_metadata.spacing[1]
	)
	vmin = data_min if arguments.vmin is None else arguments.vmin
	vmax = data_max if arguments.vmax is None else arguments.vmax
	if not np.isfinite(vmin) or not np.isfinite(vmax):
		raise ValueError("The selected density planes contain no finite values")
	if vmin >= vmax:
		padding = max(abs(vmin) * 1e-6, 1e-12)
		vmin, vmax = vmin - padding, vmax + padding

	output = arguments.output
	if output is None:
		output = Path(__file__).resolve().parent / f"rho{arguments.species}_xz.gif"
	output = output.expanduser().resolve()
	print(
		f"Creating {len(densities)} frames at y index {selected_y} "
		f"(y = {y_coordinate:g}), color range [{vmin:.7g}, {vmax:.7g}]"
	)
	create_gif(
		densities,
		timesteps,
		x,
		z,
		y_coordinate,
		reference_metadata.scalar_name,
		output,
		arguments.fps,
		vmin,
		vmax,
		arguments.cmap,
		arguments.absolute_colors,
	)
	print(f"Wrote {output}")


if __name__ == "__main__":
	main()
