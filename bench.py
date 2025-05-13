import dataclasses
import time
from collections.abc import Callable
from functools import cached_property
from typing import Any

import numpy as np
import rich.progress
import rich.table


@dataclasses.dataclass
class Record[T]:
    """A single run."""

    name: str
    duration: float
    value: T


@dataclasses.dataclass
class RecordSet[T]:
    """A set of records from the same benchmark."""

    records: list[Record[T]]

    def __post_init__(self) -> None:
        n = len({x.name for x in self.records})
        if n == 0:
            raise ValueError("Requires at least one record.")
        elif n > 1:
            raise ValueError("All records must have the same name.")

    @cached_property
    def avg(self) -> float:
        return np.mean([x.duration for x in self.records]).item()

    @cached_property
    def std(self) -> float:
        return np.std([x.duration for x in self.records]).item()

    @cached_property
    def name(self) -> str:
        return self.records[0].name

    @classmethod
    def collect(
        cls, name: str, func: Callable, *args: Any, n_repeats: int = 5, **kwargs: Any
    ) -> "RecordSet":
        records = []
        for _ in rich.progress.track(range(n_repeats), description=name):
            t0 = time.monotonic()
            result = func(*args, **kwargs)
            t1 = time.monotonic()
            records.append(Record(name, t1 - t0, result))

        return cls(records=records)

    @classmethod
    async def acollect(
        cls, name: str, func: Callable, *args: Any, n_repeats: int = 5, **kwargs: Any
    ) -> "RecordSet":
        records = []
        for _ in rich.progress.track(range(n_repeats), description=name):
            t0 = time.monotonic()
            result = await func(*args, **kwargs)
            t1 = time.monotonic()
            records.append(Record(name, t1 - t0, result))

        return cls(records=records)


@dataclasses.dataclass
class Run:
    """Results from multiple benchmarks."""

    sets: list[RecordSet]
    nbytes: int

    def summarize(self) -> rich.table.Table:
        t = rich.table.Table("Benchmark", "Duration", "Throughput (GB/s)")

        for record_set in self.sets:
            throughput = self.nbytes / record_set.avg / 1e9
            t.add_row(
                record_set.name, f"{record_set.avg:.2f} ± {record_set.std:.2f}", f"{throughput:.2f}"
            )

        return t
