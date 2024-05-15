from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable

def wrap_iterator(wrapper: Callable | type):
	def outer(func: Callable):
		def inner(*args, **kwargs):
			return wrapper(func(*args, **kwargs))
		return inner
	return outer

@dataclass(slots=True)
class Argument:
	content: str
	used: bool = False
	note: str | None = None

	@staticmethod
	@wrap_iterator(list)
	def from_args(args: Iterable[str]):
		for i in args:
			yield Argument(i, False, None)

	def __str__(self) -> str:
		return self.content

	def __repr__(self) -> str:
		return f"<'{self.content}', {'used' if self.used else 'unused'}, note={self.note}>";


@dataclass(slots=True)
class ArgumentList:
	_args: list[Argument]
	_index: int = 0

	@property
	def arguments(self):
		return self._args

	@property
	def index(self):
		return self._index

	@property
	def can_read(self):
		return self._index < len(self._args)

	@property
	def peek(self):
		return self._args[self._index]

	def next(self):
		self._index += 1
		return self._args[self._index - 1]

	def find(self, name: str, check_used_args: bool = False) -> int:
		index = -1
		for i, v in enumerate(self._args):
			if v.content == name and (check_used_args or not v.used):
				index = i
				break
		return index
	
	def find_any(self, *names: str, check_used_args: bool = False) -> int:
		"""tries to find any argument from 'names', stops after finding an argument"""
		for i in names:
			index = self.find(i, check_used_args)
			if index != -1:
				return index
		return -1

	def extract(self, name: str, check_used_args: bool = False) -> Argument | None:
		"""finds the first argument with the given name and pops it or returns none if no argument matches the name.
		
		will check used arguments if you passed True to check_used_args"""

		index = self.find(name, check_used_args)
		
		if index == -1:
			return None

		if self._index > index:
			self._index -= 1

		return self._args.pop(index)

	def extract_any(self, *names: str, check_used_args: bool = False) -> Argument | None:
		"""tries to extract any argument from 'names', stops after extracting an argument"""
		for i in names:
			arg = self.extract(i, check_used_args)
			if arg:
				return arg
		return None
	
	def extract_match(self, match_predicate: Callable[[Argument], bool]):
		"""finds the first argument satisfying the predicate and pops it or returns none if no argument matches it.

		the predicate should check if the argument is used or not"""

		for i, v in enumerate(self._args):
			if match_predicate(v):
				if self._index > i:
					self._index -= 1
				return self._args.pop(i)
		
		return None
	
	def extract_all(self, name: str):
		while self._args:
			arg = self.extract(name)
			if arg == None:
				return
			yield arg
	
	def extract_all_of(self, *names: str):
		for n in names:
			while self._args:
				arg = self.extract(n)
				if arg == None:
					return
				yield arg
	
