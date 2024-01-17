import enum
from functools import cache
import glob

from dataclasses import dataclass
from hashlib import sha1
import json
import os
import sys
from time import time_ns
from typing import Any, Iterable, Self, SupportsIndex
from pathlib import Path
import traceback

DEFUALT_PROJ_FILENAME = 'pygnu.json'
_LOG_INDENT_LEVEL = 0
_LOG_INDENT_LEVEL_S = ''

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
	
	total = sep.join(map(str, values))
	total = total.replace('\n', '\n' + _LOG_INDENT_LEVEL_S).replace('\r', '\r' + _LOG_INDENT_LEVEL_S)

	if fg is not None:
		total = f"\033[{fg}m{total}\033[0m"

	print(_LOG_INDENT_LEVEL_S + total, end=end)

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

	print_includes: bool = False # -H
	catch_typos: bool = True # -gant
	exit_on_errors: bool = False # -Wfatal-errors
	dynamicly_linkable: bool = True # --no-dynamic-linker
	print_stats: bool = True # --print-memory-usage

	libraries: LibrariesOptions
	striping: SymbolStrippingType = SymbolStrippingType.DontStrip
	include_dirs: list[str]

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

		self.print_includes = False
		self.catch_typos = True
		self.exit_on_errors = True
		self.dynamicly_linkable = True
		self.print_stats = True

		self.include_dirs = list()
		self.libraries = LibrariesOptions(dict(), list())
		self.striping = SymbolStrippingType.DontStrip

		self.assempler_args = list()
		self.linker_args = list()
		self.preprocessor_args = list()

	def create_commandline(self):
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
		
		for i in self.include_dirs:
			cl.append('-I')
			cl.append(i)

		cl.append(self.libraries.commandlet)

		cl.append(SymbolStrippingType.name(self.striping))

		
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
	def from_data(cls, data: dict[str, Any]) -> Self:
		c: Self = cls()

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
		
		# libraries

		c.libraries.directories = data.get("lib_dirs", c.libraries.directories)
		if not isinstance(c.libraries.directories, dict):
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

		return c

	def to_data(self) -> dict[str]:
		data = {}

		data["predefines"] = self.predefines.copy()

		data["optimization_lvl"] = int(self.optimization.opt_level)
		data["optimization_type"] = int(self.optimization.opt_level)

		data["standard"] = int(self.standard)

		data["warning_level"] = int(self.warnings.level)
		data["warning_pedantic"] = self.warnings.pedantic

		for i in ("print_includes", "catch_typos",
							"exit_on_errors", "dynamicly_linkable",
							"print_stats"):
			data[i] = self.__getattribute__(i)
		
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
		
		if not isinstance(c.project_dir, str | Path):
			log_err(f"invalid project directory value: \"{c.project_dir}\"")
			c.project_dir = os.getcwd()
			log(f"resseting project directory value to \"{c.project_dir}\"")
		
		if not isinstance(c.output_dir, str | Path):
			log_err(f"invalid output directory value: \"{c.output_dir}\"")
			c.output_dir = 'output'
			log(f"resseting output directory value to \"{c.project_dir}\"")
		
		if not isinstance(c.output_cache_dir, str | Path):
			log_err(f"invalid output cache directory value: \"{c.output_cache_dir}\"")
			c.output_cache_dir = c.output_dir + '/cache'
			log(f"resseting output cache directory value to \"{c.output_cache_dir}\"")
		
		if not isinstance(c.output_name, str | Path):
			log_err(f"invalid output name value: \"{c.output_name}\"")
			c.output_name = 'output'
			log(f"resseting output name value to \"{c.output_name}\"")


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
			conf = BuildConfiguration.from_data(v)
			drop_log_indent()

			log(f"successfuly parsed '{i}' build configuration", fg=LogFGColors.BrightBlack)
			c.build_configs[i] = conf
		
		return c
	
	def to_data(self):
		data = {}

		data["output_dir"] = str(self.output_dir)
		data["output_cache_dir"] = str(self.output_dir)

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
			cs.append(config.create_commandline())
			cs.append('-c')
			cs.append(f'"{i}"')
			cs.append('-o')
			
			basename = '.'.join(Path(i).resolve().name.split('.')[:-1])

			obj_filepath = self.output_cache_dir.joinpath(basename + '.o')
			object_files.append(obj_filepath)
			cs.append(f'"{obj_filepath}"')
			ls.append((' '.join(cs), i))
		
		final = []

		final.append(Project.GCC_ARG)
		final.append(config.create_commandline())

		for i in object_files:
			final.append(f'"{i}"')

		final.append('-o')
		
		output_file = ''
		if os.name == 'nt':
			output_file = self.output_dir.joinpath(f"{self.output_name}.exe")
		else:
			output_file = self.output_dir.joinpath(f"{self.output_name}.out")

		final.append(f'"{output_file}"')

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

	def build(self, confg: str):
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

def main():

	argv = sys.argv.copy()[::-1]

	while argv:
		s = argv.pop()

		match s.lower():
			case 'new':
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
						break
				defualt_prj: Project = Project.get_default_project(root_dir)
				data = defualt_prj.to_data()
				with open(proj_file, 'w') as f:
					json.dump(data, f, default=lambda x: str(x), indent=4)
				log(f"created pygnu project file at '{proj_file}'", fg=LogFGColors.BrightGreen)
			case 'build':
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

				root_dir = argv.pop() if argv else os.getcwd()

				if (root_dir[0].lower() + root_dir[1:]) in ('debug', 'release', 'prod', 'production', 'export'):
					msg = \
							f"by passing the argument '{root_dir}', " + \
							f"did you intend to build in '{root_dir}' mode? if you did, then pass '-M{root_dir}'"
					log( msg, fg=LogFGColors.Blue )
					break

				root_dir = Path(root_dir).resolve()
				proj_file = root_dir.joinpath(DEFUALT_PROJ_FILENAME)

				if not root_dir.exists():
					log(f"there is no pygnu project file at {proj_file}", fg=LogFGColors.Red)
					break


				if not proj_file.exists():
					log(f"couldn't find the pygnu project file at {proj_file}", fg=LogFGColors.Red)
					break

				log(f"building '{mode}' mode of pygnu project at '{proj_file}'")

				with open(proj_file, 'r') as f:
					data = json.load(f)
				
				log("loaded data for json-formated pygnu project from file", fg=LogFGColors.BrightBlack)
				log("deserializing pygnu project data", fg=LogFGColors.BrightBlack)

				raise_log_indent()
				project: Project = Project.from_data(data, root_dir)
				drop_log_indent()

				log("done deserializing pygnu project", fg=LogFGColors.BrightBlack)

				for i in project.build_configs:
					index = find(argv, i)
					if index != -1:
						msg = \
							f"by passing the argument '{i}', " + \
							f"did you intend to build in '{i}' mode? if you did, then pass '-M{i}'"
						log( msg, fg=LogFGColors.Blue )

				log(f"building '{mode}'", fg=LogFGColors.BrightBlack)
				
				raise_log_indent()
				if not project.build(mode):
					log("building failed!", fg=LogFGColors.Red)
				drop_log_indent()

				end_time = (time_ns() - start_time) / 1_000_000
				log(f"---- done in {end_time}ms ----", fg=LogFGColors.BrightGreen)

	# for i in d2.get_build_commands('debug'):
		
	# 	if os.system(i):
	# 		print(f"failed to excute system(\"{i}\")")

if __name__ == "__main__":
	main()
