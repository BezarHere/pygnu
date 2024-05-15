from pathlib import Path
import re
from typing import Iterable

_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"](.+?)[>\"]\s*$", re.MULTILINE)

def _check_where_is_include(include: str, include_dirs: set[Path], parent: Path):
	if parent.joinpath(include).exists():
		return parent.joinpath(include)
	
	for i in include_dirs:
		if i.joinpath(include).exists():
			return i.joinpath(include)
	return None

def get_included_filepaths(source: str, parent: Path, include_dirs: Iterable[str | Path] | None = None):
	if include_dirs == None:
		include_dirs = tuple()
	else:
		include_dirs = set(map(Path, include_dirs))

	for i in _INCLUDE_RE.finditer(source):
		p = _check_where_is_include(i[1], include_dirs, parent)
		if p != None:
			yield p