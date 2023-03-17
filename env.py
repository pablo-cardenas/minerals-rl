from dataclasses import dataclass
from dataclasses import field
from heapq import heappop
from heapq import heappush
from pprint import pprint


@dataclass(frozen=True, order=True)
class Entry:
    expected_time: float
    truck: int = field(compare=False)


@dataclass
class Location:
    current: Entry | None = None
    incoming: list[Entry] = field(default_factory=list)


@dataclass
class Truck:
    state: int | None = None
    location: tuple[int, int] | None = None


@dataclass
class Travel:
    state: int | None = None
    location: tuple[int, int] | None = None


num_queues = 3
num_trucks = 6
queues = [[Location() for _ in range(num_queues)] for _ in range(2)]
trucks = [Truck() for _ in range(num_trucks)]


def go(truck: int, queue: tuple[int, int]):
    heappush(
        queues[queue[0]][queue[1]].incoming,
        Entry(expected_time=0, truck=i),
    )
    trucks[truck].state = 0
    trucks[truck].location = queue


for i in range(num_trucks):
    go(truck=i, queue=(0, i % num_queues))

while True:
    e = min(q.incoming[0] for qs in queues for q in qs if q.incoming)
    print(e)
