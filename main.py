import enum
from functools import cache
import glob

from dataclasses import dataclass
from hashlib import sha1
import json
import os
import shutil
import sys
from time import time_ns
from typing import Any, Callable, Iterable, Self, SupportsIndex
from pathlib import Path
import traceback

DEFUALT_PROJ_FILENAME = 'pygnu.json'
_LOG_INDENT_LEVEL = 0
_LOG_INDENT_LEVEL_S = ''
_LOG_DIRTY_LAST_PRINT = True
_LOG_USE_COLOR = True

class LogFGColors(enum.IntEnum):
	Black = 30
	Red = 31
	Green = 32
	Yellow = 33
	Blue = 34
	White = 37
	BrightBlack = 90
	BrightRed = 91
	BrightGreen = 92
	BrightYellow = 93
	BrightBlue = 94
	BrightWhite = 97

def raise_log_indent():
	global _LOG_INDENT_LEVEL, _LOG_INDENT_LEVEL_S
	_LOG_INDENT_LEVEL += 1
	_LOG_INDENT_LEVEL_S = '\t' * _LOG_INDENT_LEVEL

def drop_log_indent():
	global _LOG_INDENT_LEVEL, _LOG_INDENT_LEVEL_S

	_LOG_INDENT_LEVEL -= 1
	if _LOG_INDENT_LEVEL < 0:
		_LOG_INDENT_LEVEL = 0
	_LOG_INDENT_LEVEL_S = '\t' * _LOG_INDENT_LEVEL

def log(	*values: object,
  				sep: str | None = " ",
  			  end: str | None = "\n",
					fg: LogFGColors | None = None ):
	global _LOG_DIRTY_LAST_PRINT

	total = sep.join(map(str, values))
	total = total.replace('\n', '\n' + _LOG_INDENT_LEVEL_S).replace('\r', '\r' + _LOG_INDENT_LEVEL_S)

	if _LOG_USE_COLOR and fg is not None:
		total = f"\033[{fg}m{total}\033[0m"

	if _LOG_DIRTY_LAST_PRINT:
		print( _LOG_INDENT_LEVEL_S + total, end=end)
	else:
		print( total, end=end)
	
	_LOG_DIRTY_LAST_PRINT = '\n' in end or '\r' in end

def log_err( *values: object ):
	log(*values, fg=LogFGColors.BrightRed)
	stack = traceback.extract_stack()[:-1][::-1]
	raise_log_indent()

	for i in stack:
		log(f"at '{i.name}', line {i.lineno}", fg=LogFGColors.BrightBlack)

	print(_LOG_INDENT_LEVEL_S)

	drop_log_indent()

@cache
def hash_src(src: str):
	return int.from_bytes(sha1(src.encode(), usedforsecurity=False).digest(), 'big', signed=False)

def read_data(obj: object, data: dict[str],
							property_name: str, key_name: str,
							property_validator: type | Callable[[Any], bool] | None = None,
							error: Callable[[Any], None] = None ):
	if property_validator is None:
		property_validator = type(obj.__getattribute__(property_name))
	if isinstance(property_validator, type):
		property_type = property_validator
		property_validator = lambda p: isinstance(p, property_type)
	
	if error is None:
		def _err(found_data):
			log_err("invalid data value")

@dataclass(slots=True, frozen=True)
class CommandAction:
	name: str
	desc: str
	help_desc: str = 'undocumented'

	def __call__(self, func: Callable[[Iterable[str]], Any]) -> Callable:
		func.command = self
		return func


class OptimizationType(enum.IntEnum):
	Debug = 0 # -g[x] -Og
	Speed = 1 # -O[x]
	Size = 2 # -Oz
	Space = 3 # -Os
	SpeedX = 4 # -Ofast <- maybe not conforming to the C/C++ standard
	RawDebug = 5 # -g[x] only

class OptimizationLevel(enum.IntEnum):
	NoneOptimized = 0
	Low = 1
	Medium = 2
	High = 3
	Extreme = 4

class CppStandard(enum.IntEnum):
	C98 = 0
	C11 = 1
	C14 = 2
	C17 = 3
	C20 = 4
	C2x = 5

	Cpp98 = C98 + 0xff
	Cpp11 = C11 + 0xff
	Cpp14 = C14 + 0xff
	Cpp17 = C17 + 0xff
	Cpp20 = C20 + 0xff
	Cpp2x = C2x + 0xff

	C77 = 0xffff

	@staticmethod
	def parse(name: str):
		match name:
			case "c98":
				return CppStandard.C98
			case "c11":
				return CppStandard.C11
			case "c14":
				return CppStandard.C14
			case "c17":
				return CppStandard.C17
			case "c20":
				return CppStandard.C20
			case "c2x":
				return CppStandard.C2x
			case "c++98":
				return CppStandard.Cpp98
			case "c++11":
				return CppStandard.Cpp11
			case "c++14":
				return CppStandard.Cpp14
			case "c++17":
				return CppStandard.Cpp17
			case "c++20":
				return CppStandard.Cpp20
			case "c++2x":
				return CppStandard.Cpp2x
			case _:
				return CppStandard.C77

	@staticmethod
	def name(value):
		match value:
			case CppStandard.C98:
				return "c98"
			case CppStandard.C11:
				return "c11"
			case CppStandard.C14:
				return "c14"
			case CppStandard.C17:
				return "c17"
			case CppStandard.C20:
				return "c20"
			case CppStandard.C2x:
				return "c2x"
			case CppStandard.Cpp98:
				return "c++98"
			case CppStandard.Cpp11:
				return "c++11"
			case CppStandard.Cpp14:
				return "c++14"
			case CppStandard.Cpp17:
				return "c++17"
			case CppStandard.Cpp20:
				return "c++20"
			case CppStandard.Cpp2x:
				return "c++2x"
			case _:
				return "c++17"

class SIMDType(enum.IntEnum):
	Invalid = -1
	NoSIMD = 0
	SSE = 1
	SSE2 = 2
	SSE3 = 3
	SSE4 = 4
	SSE4_1 = 5
	SSE4_2 = 6
	SSE5 = 7
	AVX = 8
	AVX2 = 10
	
	MAX = 11

	@property
	def cmd_name(self):
		match self:
			case SIMDType.Invalid:
				return 'sse2'
			case SIMDType.NoSIMD:
				return ''
			case SIMDType.SSE:
				return 'sse'
			case SIMDType.SSE2:
				return 'sse2'
			case SIMDType.SSE3:
				return 'sse3'
			case SIMDType.SSE4:
				return 'sse4'
			case SIMDType.SSE4_1:
				return 'sse4.1'
			case SIMDType.SSE4_2:
				return 'sse4.2'
			case SIMDType.SSE5:
				return 'sse5'
			case SIMDType.AVX:
				return 'avx'
			case SIMDType.AVX2:
				return 'avx2'
			
			case _:
				return 'INVALID'

	@staticmethod
	def parse(data):
		if isinstance(data, int):
			if data < 0 or data >= SIMDType.MAX:
				return SIMDType.Invalid
			return SIMDType(data)
		elif isinstance(data, str):

			lower = data.lower().replace('.', '_')
			for i,v  in SIMDType._member_map_.items():
				if i.lower() == lower:
					return SIMDType(v)
			
		return SIMDType.Invalid

class ArchitectureType(enum.IntEnum):
	...

class SymbolStrippingType(enum.IntEnum):
	DontStrip = 0
	StripDebug = 1 # -S
	StripAll = 2 # -s

	@staticmethod
	def name(value):
		match value:
			case SymbolStrippingType.DontStrip:
				return ''
			case SymbolStrippingType.StripDebug:
				return '--strip-debug'
			case SymbolStrippingType.StripAll:
				return '--strip-all'

class WarningLevel(enum.IntEnum):
	NoWarnings = 0
	NormalWarnings = 1
	AllWarnings = 2
	ExtraWarnings = 3

@dataclass(slots=True)
class Optimization:
	opt_type: OptimizationType = OptimizationType.Speed # -O / -g / -Oz
	opt_level: OptimizationLevel = OptimizationLevel.Medium

	def copy(self):
		return Optimization(self.opt_type, self.opt_level)

	@property
	def commandlet(self):
		if self.opt_level == OptimizationLevel.NoneOptimized:
			return ""

		match (self.opt_type):
			case OptimizationType.Debug:
				return f"-Og -g{min(int(self.opt_level), 4)}"
			case OptimizationType.Speed:
				return f"-O{self.opt_level}"
			case OptimizationType.Size:
				return f"-Oz"
			case OptimizationType.Space:
				return f"-Os"
			case OptimizationType.SpeedX:
				return f"-Ofast"
			case OptimizationType.RawDebug:
				return f"-g{self.opt_level}"
			case _:
				return "-O2"

@dataclass(slots=True, frozen=True)
class GlobSelector:
	_states: tuple[str, ...]

	def __call__(self, path: str | Path):
		path = str(path)
		found = set()
		for i in self._states:
			for j in glob.iglob(i, root_dir=path, recursive=True, include_hidden=True):
				if j in found:
					continue
				found.add(j)
				yield j

@dataclass(slots=True)
class WarningsOptions:
	level: WarningLevel # -W[level]
	pedantic: bool # -Wpedantic

	def copy(self):
		return WarningsOptions(self.level, self.pedantic)

	@property
	def commandlet(self):
		s = ' -Wpedantic' if self.pedantic else ''
		match self.level:
			case WarningLevel.NoWarnings:
				return '--no-warn'
			case WarningLevel.NormalWarnings:
				return '' + s
			case WarningLevel.AllWarnings:
				return '-Wall' + s
			case WarningLevel.ExtraWarnings:
				return '-Wextra' + s
		return '' + s

@dataclass(slots=True)
class LibrariesOptions:
	directories: list[str] # -L[N1] -L[N2] .. -L[Nx]
	names: list[str] # -l[N1] -l[N2] .. -l[Nx]

	def copy(self):
		return LibrariesOptions(
			self.directories.copy(), self.names.copy()
		)

	@property
	def commandlet(self):
		l = []
		for i in self.directories:
			l.append('-L')
			l.append(f'"{i}"')
		
		for i in self.names:
			l.append('-l')
			l.append(f'"{i}"')
		return ' '.join(l)

class BuildConfiguration:
	predefines: dict[str, str | None] # -D[N1] -D[N2] ... -D[Nx]
	optimization: Optimization
	standard: CppStandard = CppStandard.C17 # -std=[X] 
	warnings: WarningsOptions

	simd_type: SIMDType

	print_includes: bool = False # -H
	catch_typos: bool = True # -gant
	exit_on_errors: bool = False # -Wfatal-errors
	dynamicly_linkable: bool = True # --no-dynamic-linker
	print_stats: bool = True # --print-memory-usage

	libraries: LibrariesOptions
	striping: SymbolStrippingType = SymbolStrippingType.DontStrip
	include_dirs: list[str]

	# TODO implement
	# excluded_files: set[FILENAME/GLOB SELECTOR]
	# output_type: dict[FILENAME, OUTPUT_TYPE]

	assempler_args: list[str]
	linker_args: list[str]
	preprocessor_args: list[str]

	# simple boolean properties
	FLAG_PROPERTIES = \
		'print_includes', 'catch_typos', \
		'exit_on_errors', 'dynamicly_linkable', \
		'print_stats'
	
	# copy on write properties
	SIMPLE_PROPERTIES = FLAG_PROPERTIES + ( 'striping', 'standard' )

	# properties that can by copied by calling 'x.copy()'
	COPYCALL_PROPERTIES = \
		'predefines', 'include_dirs', 'assempler_args', 'linker_args', 'preprocessor_args', \
		'optimization', 'warnings', 'libraries'

	def __init__(self) -> None:
		self.predefines = dict()
		self.optimization = Optimization()
		self.standard = CppStandard.C17
		self.warnings = WarningsOptions(WarningLevel.NormalWarnings, False)

		self.simd_type = SIMDType.SSE

		self.print_includes = False
		self.catch_typos = True
		self.exit_on_errors = True
		self.dynamicly_linkable = True
		self.print_stats = True

		self.include_dirs = list()
		self.libraries = LibrariesOptions(list(), list())
		self.striping = SymbolStrippingType.DontStrip

		self.assempler_args = list()
		self.linker_args = list()
		self.preprocessor_args = list()

	def create_commandline(self, mid_section: str):
		cl = []
		for i, v in self.predefines.items():
			if v is not None:
				cl.append(f"-D{i}=\"{v}\"")
			else:
				cl.append(f"-D{i}")
		
		cl.append(self.optimization.commandlet)
		cl.append(f"-std={CppStandard.name(self.standard)}")
		cl.append(self.warnings.commandlet)

		if self.print_includes:
			cl.append('-H')
		# if self.catch_typos:
		# 	cl.append('-gant')
		if self.exit_on_errors:
			cl.append('-Wfatal-errors')
		if not self.dynamicly_linkable:
			cl.append('--no-dynamic-linker')
		# if self.print_stats:
		# 	cl.append('--print-memory-usage')
		
		if self.simd_type > SIMDType.NoSIMD:
			cl.append(f'-m{self.simd_type.cmd_name}')

		for i in self.include_dirs:
			cl.append('-I')
			cl.append(i)


		cl.append(SymbolStrippingType.name(self.striping))

		cl.append(mid_section)

		cl.append(self.libraries.commandlet)
		
		# remove empty commandlets
		return ' '.join(i for i in cl if i)

	def copy(self):
		c = BuildConfiguration()

		for i in BuildConfiguration.SIMPLE_PROPERTIES:
			c.__setattr__(i, self.__getattribute__(i))

		for i in BuildConfiguration.COPYCALL_PROPERTIES:
			c.__setattr__(i, self.__getattribute__(i).copy())

		return c

	@classmethod
	def from_data(cls, data: dict[str, Any]):
		c: Self = cls()
		# if there is an invalid value that has been defaulted
		reset_anything: bool = False 

		# predefines

		c.predefines = data.get("predefines", dict())
		if isinstance(c.predefines, list | tuple | set | frozenset):
			pred = {}
			for i in c.predefines:
				pred[i] = None
			c.predefines = pred
		elif not isinstance(c.predefines, dict):
			raise ValueError(f"invalid predefines value: \"{c.predefines}\"")	

		# optimization

		c.optimization.opt_level = data.get("optimization_lvl", c.optimization.opt_level)
		if not isinstance(c.optimization.opt_level, int | OptimizationLevel):
			raise ValueError(f"invalid optimization.opt_level value: \"{c.optimization.opt_level}\"")	

		c.optimization.opt_level = OptimizationLevel(c.optimization.opt_level)

		c.optimization.opt_type = data.get("optimization_type", c.optimization.opt_type)
		if not isinstance(c.optimization.opt_type, int | OptimizationType):
			raise ValueError(f"invalid optimization.opt_type value: \"{c.optimization.opt_type}\"")	

		c.optimization.opt_type = OptimizationType(c.optimization.opt_type)

		# standard

		c.standard = data.get("standard", c.standard)
		if not isinstance(c.standard, int | CppStandard):
			if isinstance(c.standard, str):
				name = c.standard
				c.standard = CppStandard.parse(name.lower())

				if c.standard == CppStandard.C77:
					log_err(f"no standard version exists with the value: \"{name}\"")
					c.standard = CppStandard.C17
					log(f"default standard version to C17", fg=LogFGColors.Green)
			else:
				raise ValueError(f"invalid standard version value: \"{c.standard}\"")	

		c.standard = CppStandard(c.standard)

		# warnings

		c.warnings.level = data.get("warning_level", c.warnings.level)
		if not isinstance(c.warnings.level, int | WarningLevel):
			raise ValueError(f"invalid warning level value: \"{c.warnings.level}\"")	

		c.warnings.level = WarningLevel(c.warnings.level)

		c.warnings.pedantic = data.get("warning_pedantic", c.warnings.pedantic)
		if not isinstance(c.warnings.pedantic, bool):
			raise ValueError(f"invalid warning pedantic value: \"{c.warnings.pedantic}\"")	

		c.warnings.pedantic = bool(c.warnings.pedantic)

		# flags
		for i in ( "print_includes", "catch_typos",
							"exit_on_errors", "dynamicly_linkable",
							"print_stats" ):
			attr = c.__getattribute__(i)
			attr_type = type(attr)
			c.__setattr__(i, data.get(i, attr))
			if not isinstance(c.__getattribute__(i), attr_type):
				raise ValueError(f"invalid {i} value: \"{c.__getattribute__(i)}\"")	

			c.__setattr__(i, attr_type(c.__getattribute__(i)))
		
		
		# simd

		simd_raw_value = data.get('simd_type')
		simd_type = SIMDType.parse(simd_raw_value)

		if simd_type == SIMDType.Invalid:
			log_err(f"invalid simd type value: \"{simd_raw_value}\"")
			simd_type = None

		if simd_type is None:
			log(f"default simd type value to \"{c.simd_type.name.lower()}\"", fg=LogFGColors.Green)
			reset_anything = True
		else:
			c.simd_type = simd_type

		# libraries

		c.libraries.directories = data.get("lib_dirs", c.libraries.directories)
		if not isinstance(c.libraries.directories, list | tuple):
			raise ValueError(f"invalid library directories value: \"{c.libraries.directories}\"")	

		c.libraries.directories = list(c.libraries.directories)

		c.libraries.names = data.get("lib_names", c.libraries.names)
		if not isinstance(c.libraries.names, list | tuple):
			raise ValueError(f"invalid library names value: \"{c.libraries.names}\"")	

		c.libraries.names = list(c.libraries.names)

		# includes

		c.include_dirs = data.get("include_dirs", c.include_dirs)
		if not isinstance(c.include_dirs, list | tuple):
			raise ValueError(f"invalid include directories value: \"{c.include_dirs}\"")	

		c.include_dirs = list(c.include_dirs)

		# commands

		c.assempler_args = data.get("assempler_args", c.assempler_args)
		if not isinstance(c.assempler_args, list | tuple):
			raise ValueError(f"invalid assempler arguments value: \"{c.assempler_args}\"")	

		c.assempler_args = list(c.assempler_args)

		c.linker_args = data.get("linker_args", c.linker_args)
		if not isinstance(c.linker_args, list | tuple):
			raise ValueError(f"invalid linker arguments value: \"{c.linker_args}\"")	

		c.linker_args = list(c.linker_args)

		c.preprocessor_args = data.get("preprocessor_args", c.preprocessor_args)
		if not isinstance(c.preprocessor_args, list | tuple):
			raise ValueError(f"invalid preprocessor arguments value: \"{c.preprocessor_args}\"")	

		c.preprocessor_args = list(c.preprocessor_args)

		return c, reset_anything

	def to_data(self) -> dict[str]:
		data = {}

		data["predefines"] = self.predefines.copy()

		data["optimization_lvl"] = int(self.optimization.opt_level)
		data["optimization_type"] = int(self.optimization.opt_level)

		data["standard"] = CppStandard.name(self.standard)

		data["warning_level"] = int(self.warnings.level)
		data["warning_pedantic"] = self.warnings.pedantic

		for i in ("print_includes", "catch_typos",
							"exit_on_errors", "dynamicly_linkable",
							"print_stats"):
			data[i] = self.__getattribute__(i)
		
		data["simd_type"] = self.simd_type.cmd_name
		
		data["include_dirs"] = self.include_dirs.copy()
		data["lib_dirs"] = self.libraries.directories.copy()
		data["lib_names"] = self.libraries.names.copy()

		data["assempler_args"] = self.assempler_args.copy()
		data["linker_args"] = self.linker_args.copy()
		data["preprocessor_args"] = self.preprocessor_args.copy()
		
		return data


@dataclass(slots=True)
class Project:
	GCC_ARG = 'gcc'
	GPP_ARG = 'g++'
	BUILD_HASH_FILENAME = 'build.hash'

	project_dir: Path
	output_dir: Path
	output_cache_dir: Path
	output_name: str
	build_configs: dict[str, BuildConfiguration]
	source_selector: GlobSelector = GlobSelector(( "**/*.c",
																										 "**/*.cpp",
																										 "**/*.cc",
																										 "**/*.cxx" ))
	
	def replace_macros(self, path: str):
		return path \
			.replace('$(OutputDir)', str(self.output_dir)) \
			.replace('$(OutputCacheDir)', str(self.output_cache_dir)) \
			.replace('$(ProjectDir)', str(self.project_dir)) \

	@staticmethod
	def from_data(data: dict[str], project_dir: Path):
		
		c = Project(
			project_dir,
			data.get("output_dir"),
			data.get("output_cache_dir"),
			data.get("output_name"),
			dict(),
			)
		needs_resaving = False
		
		if not isinstance(c.project_dir, str | Path):
			log_err(f"invalid project directory value: \"{c.project_dir}\"")

			c.project_dir = os.getcwd()
			needs_resaving = True

			log(f"resseting project directory value to \"{c.project_dir}\"", fg=LogFGColors.Green)
		
		if not isinstance(c.output_dir, str | Path):
			log_err(f"invalid output directory value: \"{c.output_dir}\"")

			c.output_dir = 'output'
			needs_resaving = True

			log(f"resseting output directory value to \"{c.project_dir}\"", fg=LogFGColors.Green)
		
		if not isinstance(c.output_cache_dir, str | Path):
			log_err(f"invalid output cache directory value: \"{c.output_cache_dir}\"")

			c.output_cache_dir = c.output_dir + '/cache'
			needs_resaving = True

			log(f"resseting output cache directory value to \"{c.output_cache_dir}\"", fg=LogFGColors.Green)
		
		if not isinstance(c.output_name, str | Path):
			log_err(f"invalid output name value: \"{c.output_name}\"")

			c.output_name = 'output'
			needs_resaving = True

			log(f"resseting output name value to \"{c.output_name}\"", fg=LogFGColors.Green)


		c.project_dir = Path(c.project_dir).absolute()
		c.output_dir = Path(c.replace_macros(str(c.output_dir))).absolute()
		c.output_cache_dir = Path(c.replace_macros(str(c.output_cache_dir))).absolute()

		# source selector

		selectors = data.get("source_selectors")
		if not isinstance(selectors, list | tuple):
			raise ValueError(f"invalid source selectors value: \"{selectors}\"")

		selectors = tuple(selectors)

		for i, v in enumerate(selectors):
			if not isinstance(v, str):
				raise ValueError(f"invalid source selector at index {i}, with value: \"{v}\"")

		c.source_selector = GlobSelector(selectors)

		configs = data.get("build_configurations")

		if not isinstance(configs, dict):
			raise ValueError(f"invalid build configurations value: \"{configs}\"")
		
		for i, v in configs.items():
			if not isinstance(i, str):
				raise ValueError(f"invalid build configuration name: \"{i}\"")
			
			if not isinstance(v, dict):
				raise ValueError(f"'{i}' has an invalid build configuration value: \"{v}\"")
			
			log(f"parsing '{i}' build configuration", fg=LogFGColors.BrightBlack)

			raise_log_indent()
			conf, reset_anything = BuildConfiguration.from_data(v)
			drop_log_indent()

			if reset_anything:
				needs_resaving = True

			log(f"successfuly parsed '{i}' build configuration", fg=LogFGColors.BrightBlack)
			c.build_configs[i] = conf
		
		return c, needs_resaving
	
	def to_data(self):
		data = {}

		if self.output_dir.is_relative_to(self.project_dir):
			data["output_dir"] = str(self.output_dir.relative_to(self.project_dir))
		elif self.output_dir.absolute() == self.project_dir.absolute():
			data["output_dir"] = '.'
		else:
			data["output_dir"] = str(self.output_dir.absolute())
		
		if self.output_cache_dir.is_relative_to(self.project_dir):
			data["output_cache_dir"] = str(self.output_cache_dir.relative_to(self.project_dir))
		elif self.output_cache_dir.absolute() == self.project_dir.absolute():
			data["output_cache_dir"] = '.'
		else:
			data["output_cache_dir"] = str(self.output_cache_dir.absolute())

		data["output_name"] = str(self.output_name)

		data["source_selectors"] = self.source_selector._states

		
		data["build_configurations"] = {}
		for i, v in self.build_configs.items():
			data["build_configurations"][i] = v.to_data()

		return data

	def gather_source_files(self):
		return self.source_selector(self.project_dir)

	def get_build_commands(self, config_name: str):
		if not config_name in self.build_configs:
			log(f"build config '{config_name}' doesn't exist!")
			return []
		
		ls = []
		config = self.build_configs[config_name]

		object_files = []

		for i in self.gather_source_files():
			cs = []
			cs.append(Project.GCC_ARG)
			
			
			basename = '.'.join(Path(i).resolve().name.split('.')[:-1])

			obj_filepath = self.output_cache_dir.joinpath(basename + '.o')
			object_files.append(obj_filepath)

			cs.append(config.create_commandline(f'-c "{i}" -o "{obj_filepath}"'))
			ls.append((' '.join(cs), i))
		
		
		output_file = ''
		if os.name == 'nt':
			output_file = self.output_dir.joinpath(f"{self.output_name}.exe")
		else:
			output_file = self.output_dir.joinpath(f"{self.output_name}.out")
		
		obj_file_paths = ' '.join(f'"{i}"' for i in object_files)

		final = []

		final.append(Project.GCC_ARG)

		final.append(config.create_commandline(f'{obj_file_paths} -o "{output_file}"'))

		ls.append((' '.join(final), output_file))

		return ls

	@staticmethod
	@cache
	def get_default_project(project_path: Path):
		debug_build = BuildConfiguration()
		debug_build.optimization = Optimization(OptimizationType.Debug)
		debug_build.warnings = WarningsOptions(WarningLevel.AllWarnings, False)
		debug_build.predefines = dict(_DEBUG=None)

		release_build = debug_build.copy()
		release_build.optimization = Optimization(OptimizationType.Speed, OptimizationLevel.Extreme)
		release_build.predefines = dict(NDEBUG=None,_RELEASE=None)

		return Project(
			project_path, Path(), Path(), 'output',
			dict(debug=debug_build, release=release_build) )

	def build(self, confg: str, verbose: bool = False):
		"""returns weather the build succesful"""

		if not confg in self.build_configs:
			log(f"project: no config with name '{confg}'", fg=LogFGColors.BrightYellow)
			configs_str = []
			for i in self.build_configs:
				configs_str.append(i)
			
			last_name = f" and '{configs_str[-1]}'" if len(configs_str) > 1 else ''
			other_names = ', '.join(map(lambda s: f"'{s}'", configs_str[:-1]))
			log( f"available configs are {other_names}" + last_name, fg=LogFGColors.BrightBlue )
			return False
		
		if not self.project_dir.exists():
			log_err(f"invalid project directory: no directory found at '{self.project_dir}'")
			return False
		
		if not self.output_dir.exists():
			log(f"created output directory at '{self.output_dir}'", fg=LogFGColors.Green)
			self.output_dir.mkdir(exist_ok=True, parents=True)
		
		if not self.output_cache_dir.exists():
			log(f"created output cache directory at '{self.output_cache_dir}'", fg=LogFGColors.Green)
			self.output_cache_dir.mkdir(exist_ok=True, parents=True)

		for cmd, file in self.get_build_commands(confg):
			file = Path(file).resolve()

			if verbose:
				log(f"VERBOSE: executing command: \"", end='', fg=LogFGColors.BrightBlack)
				log(cmd, end='', fg=LogFGColors.BrightBlue)
				log("\" on file \"", end='', fg=LogFGColors.BrightBlack)
				log(file, end='', fg=LogFGColors.BrightBlue)
				log('"')
			
			if os.system(cmd):
				log(f"failed to execute system cmd: \"{cmd}\"", fg=LogFGColors.BrightRed)
			else:
				log(f"compiled '{file.relative_to(self.project_dir)}'", fg=LogFGColors.BrightBlack)
		return True

def find(it: Iterable, value) -> SupportsIndex:
	try:
		f = it.index(value)
	except ValueError:
		return -1
	except Any:
		raise
	return f


def commands_table():
	def wraper(cls: type):

		command_funcs = {}

		for i, v in cls.__dict__.items():
			# no internal states
			if i.startswith('__') and i.endswith('__'):
				continue
			
			# only static methods
			if not isinstance(v, staticmethod):
				continue
			
			if not isinstance(v.__getattribute__('command'), CommandAction):
				continue

			command_funcs[i] = v

		cls.commands = {v.command.name: v for i, v in command_funcs.items()}
		cls.func_mapped_commands = command_funcs

		return cls
	return wraper


@commands_table()
class Commands:
	__new__ = None
	_new_help_desc = ""
	_build_help_desc = ""
	_edit_help_desc = ""

	#* CommandAction should always come before the staticmethod decorator

	@CommandAction("new", "creates a new pygnu project", help_desc=_new_help_desc)
	@staticmethod
	def new(argv: list[str]):
		overwrite = find(argv, '--overwrite')

		if overwrite != -1:
			argv.pop(overwrite)
			overwrite = True
		else:
			overwrite = False

		root_dir = argv.pop() if argv else os.getcwd()
		root_dir = Path(root_dir).resolve()
		if not root_dir.exists():
			root_dir.mkdir(exist_ok=True, parents=True)

		proj_file = root_dir.joinpath(DEFUALT_PROJ_FILENAME)

		if proj_file.exists():
			if overwrite:
				log(f"overwriten file while creating a new pygnu project at '{proj_file}'",
						fg=LogFGColors.Yellow)
			else:
				log(f"Can't create a new pygnu project file at '{proj_file}': file already exists",
						fg=LogFGColors.BrightRed)
				return False
		
		defualt_prj: Project = Project.get_default_project(root_dir)
		data = defualt_prj.to_data()

		with open(proj_file, 'w') as f:
			json.dump(data, f, default=lambda x: str(x), indent=4)
		
		log(f"created pygnu project file at '{proj_file}'", fg=LogFGColors.BrightGreen)
		return True

	@CommandAction(name='build', desc='build the pygnu project', help_desc=_build_help_desc)
	@staticmethod
	def build(argv: list[str]):
		start_time = time_ns()
		mode = 'debug'

		if True:
			mode_index = -1
			for i, v in enumerate(argv):
				if v[:2] == '-M':
					mode = v[2:]
					mode_index = i
			
			if mode_index != -1:
				argv.pop(mode_index)

		verbose: SupportsIndex | bool = find(argv, '-v')
		if verbose != -1:
			argv.pop(verbose)
			verbose = True
		else:
			verbose = False

		root_dir = argv.pop() if argv else os.getcwd()

		if (root_dir[0].lower() + root_dir[1:]) in ('debug', 'release', 'prod', 'production', 'export'):
			msg = \
					f"by passing the argument '{root_dir}', " + \
					f"did you intend to build in '{root_dir}' mode? if you did, then pass '-M{root_dir}'"
			log( msg, fg=LogFGColors.Blue )
			return False

		root_dir = Path(root_dir).resolve()
		proj_file = root_dir.joinpath(DEFUALT_PROJ_FILENAME)

		if not root_dir.exists():
			log(f"there is no pygnu project file at {proj_file}", fg=LogFGColors.Red)
			return False


		if not proj_file.exists():
			log(f"couldn't find the pygnu project file at {proj_file}", fg=LogFGColors.Red)
			return False

		log(f"building '{mode}' mode of pygnu project at '{proj_file}'", fg=LogFGColors.BrightBlue)

		with open(proj_file, 'r') as f:
			data = json.load(f)
		
		log("loaded data for json-formated pygnu project from file", fg=LogFGColors.BrightBlack)
		log("deserializing pygnu project data", fg=LogFGColors.BrightBlack)

		raise_log_indent()
		project, project_needs_resaving = Project.from_data(data, root_dir)
		drop_log_indent()

		log("done deserializing pygnu project", fg=LogFGColors.BrightBlack)

		if project_needs_resaving:
			log("resaving project to apply fixes", fg=LogFGColors.Yellow)
			shutil.copy(proj_file, str(proj_file) + ".last")
			with open(proj_file, 'w') as f:
				json.dump(project.to_data(), f, indent=4, default=str)

		# log("checking paramters")

		for v in project.build_configs:
			index = find(argv, v)
			if index != -1:
				msg = \
					f"by passing the argument '{v}', " + \
					f"did you intend to build in '{v}' mode? if you did, then pass '-M{v}'"
				log( msg, fg=LogFGColors.Blue )

		log(f"building '{mode}'", fg=LogFGColors.BrightBlack)
		
		raise_log_indent()
		if not project.build(mode, verbose):
			log("building failed!", fg=LogFGColors.Red)
		drop_log_indent()

		end_time = (time_ns() - start_time) / 1_000_000
		log(f"---- done in {end_time}ms ----", fg=LogFGColors.BrightGreen)
		return True

	@CommandAction(name='edit', desc='edits the given pygnu project', help_desc=_edit_help_desc)
	@staticmethod
	def edit(argv: list[str]):
		...

	@CommandAction(name='help', desc='shows help on given command or general help if non is given',
								 help_desc='no help on help? :[')
	@staticmethod
	def help(argv: list[str]):
		for i, v in Commands.commands.items():
			ca: CommandAction = v.command

			log(i, fg=LogFGColors.BrightBlue)
			raise_log_indent()
			log(ca.desc + '\n')
			log(ca.help_desc, fg=LogFGColors.BrightBlack)
			drop_log_indent()
		return True

def main():

	argv = sys.argv.copy()[1:]
	command: Callable | None = None
	command_index = -1

	for i, v in enumerate(argv):
		if not v:
			continue

		if v[0] == '-' or v[0] == '/':
			global _LOG_USE_COLOR

			if v == '--no-color':
				_LOG_USE_COLOR = False
			elif v == '--color':
				_LOG_USE_COLOR = True
			
			continue

		if v in Commands.commands:

			if command is not None:
				log_err(f"found unexpected argument: \"{v}\"")
				continue

			if i != 0:
				log(f"found run mode '{v}', but it wasn't the first argument, "
						 "please put the run mode in the first argument", fg=LogFGColors.Yellow)

			command = Commands.commands[v]
			command_index = i
			break

	if command_index != -1:
		argv.pop(command_index)

	if command is None:
		log_err("no run command was found! please refer to help")
		exit(1)

	command(argv)
				

	# for i in d2.get_build_commands('debug'):
		
	# 	if os.system(i):
	# 		print(f"failed to excute system(\"{i}\")")

if __name__ == "__main__":
	main()
