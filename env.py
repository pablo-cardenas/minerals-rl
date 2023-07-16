from dataclasses import dataclass
from dataclasses import field
from heapq import heappop
from heapq import heappush
from pprint import pprint
import numpy as np
from numpy.random import default_rng

@dataclass
class Place:
    queue: list[int] = field(default_factory=list)


@dataclass
class Truck:
    travel_origin: tuple[int, int] | None = None
    travel_destination: tuple[int, int] | None = None
    travel_remaining: float = 0
    work_place: tuple[int, int] | None = None
    work_remaining: float = 0
    status : str = "work"


rng = default_rng(42)
num_trucks = 6
num_places =(3, 3)
list_place = [[Place() for _ in range(num_places[i])] for i in range(2)]
list_truck = [Truck() for _ in range(num_trucks)]
capacity = 20 # t
distance = 5 # km
travel_speed = 20 / 60 # km/min
work_speed = 20 # t/min
work_sigma = 1 # t
travel_sigma = 1


# Initialization
for i_truck in range(num_trucks):
    # truck i travel to shovel i%num_places
    list_truck[i_truck].travel_destination = (0, i_truck % num_places[0])
    list_truck[i_truck].travel_remaining = 2
    list_truck[i_truck].status = "travel"


for _ in range(20):
    list_remaining = []
    for i_truck in range(num_trucks):
        truck = list_truck[i_truck]
        if truck.status == 'work':
            mean = truck.work_remaining / work_speed
            scale = truck.work_remaining ** 2 / work_sigma ** 2
        elif truck.status == 'travel':
            mean = truck.travel_remaining / travel_speed
            scale = truck.travel_remaining ** 2 / travel_sigma ** 2
        remaining = 0 if mean == 0 else rng.wald(mean, scale)

        list_remaining.append(remaining)
        
    id_truck_first = min(range(num_trucks), key=list_remaining.__getitem__)
    remaining_first = list_remaining[id_truck_first]
    print(f"{id_truck_first=} {remaining_first=}")
    truck_first = list_truck[id_truck_first]

    # update the others trucks
    for i_truck in range(num_trucks):
        if i_truck == id_truck_first:
            continue
        
        truck = list_truck[i_truck]
        if truck.status == 'work':
            while True:
                w_t = rng.normal(0, np.sqrt(remaining_first))
                displacement = work_speed * remaining_first + work_sigma * w_t
                if truck.work_remaining >= displacement:
                    truck.work_remaining -= displacement
                    break
        elif truck.status == 'travel':
            while True:
                w_t = rng.normal(0, np.sqrt(remaining_first))
                displacement = travel_speed * remaining_first + travel_sigma * w_t
                if truck.travel_remaining >= displacement:
                    truck.travel_remaining -= displacement
                    break
        elif truck.status == 'queue':
            pass


    # update the first_truck
    if truck_first.status == 'work':
        action = int(input(f'A donde mando el camion {id_truck_first}: '))

        truck_first.status = 'travel'
        truck_first.travel_origin = truck_first.travel_destination
        truck_first.travel_destination = (1 - truck_first.travel_destination[0], action)
        truck_first.travel_remaining = distance

        if list_place[truck_first.travel_destination[0]][truck_first.travel_destination[1]].queue:
            i_truck_next = list_place[truck.travel_destination].queue.pop(0)
            truck_next = list_truck[i_truck_next]
            truck_next.status = 'work'
            truck_next.work_place = truck_next.travel_destination
            truck_next.work_remaining = capacity

    elif truck_first.status == 'travel':
        if list_place[truck_first.travel_destination[0]][ truck_first.travel_destination[1]].queue:
            truck_first.status = 'queue'
            list_place[truck.travel_destination]
        else:
            truck_first.status = "work"
            truck_first.work_place = truck_first.travel_destination
            truck_first.work_remaining = capacity
