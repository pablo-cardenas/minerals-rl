from dataclasses import dataclass
from dataclasses import field
from heapq import heappop
from heapq import heappush
from pprint import pprint
import numpy as np
from numpy.random import default_rng

@dataclass
class Place:
    current_truck: int | None = None
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
travel_speed = 30 / 60 # km/min
work_speed = 20 # t/min
work_sigma = 0 # t
travel_sigma = 0


# Initialization
for i_truck in range(num_trucks):
    # truck i travel to shovel i%num_places
    list_truck[i_truck].travel_destination = (0, i_truck % num_places[0])
    list_truck[i_truck].travel_origin = (1, i_truck % num_places[0])
    list_truck[i_truck].travel_remaining = 2
    list_truck[i_truck].status = "travel"


for _ in range(20):
    # Encontrar la lista de tiempos faltantes
    list_time_remaining = []
    for i_truck in range(num_trucks):
        truck = list_truck[i_truck]
        if truck.status == 'work':
            remaining = truck.work_remaining
            speed = work_speed
            sigma = work_sigma
        elif truck.status == 'travel':
            remaining = truck.travel_remaining
            speed = travel_speed
            sigma = travel_sigma

        mean = remaining / speed

        try:
            scale = remaining ** 2 / sigma ** 2
        except ZeroDivisionError:
            time_remaining = mean
        else:
            if mean == 0:
                time_remaining = 0
            else:
                time_remaining = rng.wald(mean, scale)


        list_time_remaining.append(time_remaining)
        
    # Encontrar el camion que hace el evento m'as proximo
    id_truck_first = min(range(num_trucks), key=list_time_remaining.__getitem__)
    remaining_first = list_time_remaining[id_truck_first]
    print()
    print(f"{list_time_remaining}")
    print(f"{[truck.status for truck in list_truck]}")
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

        place = list_place[truck_first.travel_destination[0]][truck_first.travel_destination[1]]
        place.current_truck = None

        if place.queue:
            i_truck_next = place.queue.pop(0)
            truck_next = list_truck[i_truck_next]
            truck_next.status = 'work'
            truck_next.work_place = truck_next.travel_destination
            truck_next.work_remaining = capacity
            place.current_truck = i_truck_next

    elif truck_first.status == 'travel':
        place = list_place[truck_first.travel_destination[0]][truck_first.travel_destination[1]]
        if place.current_truck is not None:
            truck_first.status = 'queue'
            place.queue.append(id_truck_first)
        else:
            truck_first.status = "work"
            truck_first.work_place = truck_first.travel_destination
            truck_first.work_remaining = capacity
            place.current_truck = id_truck_first
