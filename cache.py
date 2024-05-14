
from dataclasses import dataclass
from pathlib import Path




@dataclass(slots=True, frozen=True)
class BuildItemCache:
	source: Path
	target: Path

class BuildCache:
	items 
