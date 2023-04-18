from dataclasses import dataclass
from dataclasses import field
from heapq import heappop
from heapq import heappush
from pprint import pprint


@dataclass
class Place:
    current: int | None = None
    incoming: list[int] = field(default_factory=list)


@dataclass
class Truck:
    fromPlace: tuple[int, int] | None = None
    toPlace: tuple[int, int] | None = None
    departure_time: float = 0
    work_remaining: float = 0
    arrival_time: float = 0


num_places = 3
num_trucks = 6
places = [[Place() for _ in range(num_places)] for _ in range(2)]
trucks = [Truck() for _ in range(num_trucks)]


def go(truck: int, place: tuple[int, int], departure_time: float):
    # Append truck to incomming 
    places[place[0]][place[1]].incoming.append(truck)

    trucks[truck].fromPlace = trucks[truck].toPlace
    trucks[truck].toPlace = place
    trucks[truck].departure_time = departure_time
    trucks[truck].arrival_time = departure_time + 5



# Initialization
for i in range(num_trucks):
    # truck i go to shovel i%num_places
    go(truck=i, place=(0, i % num_places), departure_time = 0)

for _ in range(10):
    id_truck = min(range(num_trucks), key=lambda i:trucks[i].arrival_time)
    truck = trucks[id_truck]

    pprint(trucks)
    action = int(input(f'A donde mando el camion {id_truck}: '))
    go(id_truck, (1 - truck.toPlace[0], action), truck.arrival_time)
