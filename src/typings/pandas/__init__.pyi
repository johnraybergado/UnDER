# typings/pandas/__init__.pyi
from typing import Any, Iterator

class Series:
    def __getitem__(self, key: str) -> Any: ...

class DataFrame:
    def iterrows(self) -> Iterator[tuple[Any, Series]]: ...

def read_csv(
    filepath_or_buffer: str,
    *,
    sep: str = ...,
    delimiter: str | None = ...,
    header: int | str | None = ...,
    **kwargs: Any,
) -> DataFrame: ...