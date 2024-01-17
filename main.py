import enum
import glob

from dataclasses import dataclass
import json
import os
import sys
from typing import Any, Self
from pathlib import Path
from glassy.utils import Tape
from sklearn.decomposition import dict_learning


DEFUALT_PROJ_FILENAME = 'pygnu.json'

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

	@property
	def commandlet(self):
		if self.opt_level == OptimizationLevel.NoneOptimized:
			return ""

		match (self.opt_type):
			case OptimizationType.Debug:
				return f"-Og -g{self.opt_level}"
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

	include_dirs: list[str]
	libraries: LibrariesOptions
	striping: SymbolStrippingType = SymbolStrippingType.DontStrip

	assempler_args: list[str]
	linker_args: list[str]
	preprocessor_args: list[str]

	FLAG_PROPERTIES = \
		'print_includes', 'catch_typos', \
		'exit_on_errors', 'dynamicly_linkable', \
		'print_stats'
	
	# copy on write properties
	SIMPLE_PROPERTIES = FLAG_PROPERTIES + ( 'striping', 'standard' )

	# properties that can by copied by calling 'x.copy()'
	COPYCALL_PROPERTIES = FLAG_PROPERTIES + ( 'striping', 'standard' )

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

		c.predefines = self.predefines.copy()
		c.optimization = Optimization(self.optimization.opt_type, self.optimization.opt_level)
		c.warnings = WarningsOptions(self.warnings.level, self.warnings.pedantic)
		c.standard = self.standard

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

	project_dir: Path
	output_dir: Path
	output_cache_dir: Path
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
			dict(),
			)
		
		if not isinstance(c.project_dir, str | Path):
			raise ValueError(f"invalid project directory value: \"{c.project_dir}\"")
		
		if not isinstance(c.output_dir, str | Path):
			raise ValueError(f"invalid output directory value: \"{c.output_dir}\"")
		
		if not isinstance(c.output_cache_dir, str | Path):
			raise ValueError(f"invalid output cache directory value: \"{c.output_cache_dir}\"")


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
			
			print(f"parsing '{i}' build configuration:")
			conf = BuildConfiguration.from_data(v)
			print(f"successfuly parsed '{i}' build configuration")
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
			print(f"build config '{config_name}' doesn't exist!")
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
			ls.append(' '.join(cs))
		
		final = []

		final.append(Project.GCC_ARG)
		final.append(config.create_commandline())

		for i in object_files:
			final.append(f'"{i}"')

		final.append('-o')
		
		if os.name == 'nt':
			final.append(f'"{self.project_dir.joinpath("output.exe")}"')
		else:
			final.append(f'"{self.project_dir.joinpath("output.out")}"')

		ls.append(' '.join(final))

		return ls

	@staticmethod
	def get_default_project(project_path: Path):
		debug_build = BuildConfiguration()
		debug_build.optimization = Optimization(OptimizationType.Debug)
		debug_build.warnings = WarningsOptions(WarningLevel.AllWarnings, False)
		debug_build.predefines = dict(_DEBUG=None)

		return Project( project_path, Path(), Path(), dict() )

def main():

	argv = sys.argv.copy()[::-1]

	while argv:
		s = argv.pop()

		match s.lower():
			case 'new':
				root_dir = argv.pop() if argv else os.getcwd()
				root_dir = Path(root_dir).resolve()
				if not root_dir.exists():
					root_dir.mkdir(exist_ok=True, parents=True)

				new_file = root_dir.joinpath(DEFUALT_PROJ_FILENAME)
				if new_file.exists():
					print(f"Can not create a new pygnu project file at '{new_file}': file already exists")
					break
				
				

			case 'build':
				

	input('d')
	c = BuildConfiguration()
	d = Project("","","", dict(debug=c))

	with open("project1.json", 'w') as f:
		json.dump(d.to_data(), f, default=lambda o: str(o), indent=4)

	with open("project1.json", 'r') as f:
		json_data = json.load(f)

	d2 = Project.from_data(json_data, "")

	with open("project2.json", 'w') as f:
		json.dump(d2.to_data(), f, default=lambda o: str(o), indent=4)
	
	print(tuple(d2.gather_source_files()))
	# print(d2.get_build_commands('debug'))

	for i in d2.get_build_commands('debug'):
		
		if os.system(i):
			print(f"failed to excute system(\"{i}\")")

if __name__ == "__main__":
	main()
