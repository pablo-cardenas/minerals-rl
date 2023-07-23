from dataclasses import dataclass
from dataclasses import field
from heapq import heappop
from heapq import heappush
from pprint import pprint
from typing import Optional

import gymnasium as gym
import numpy as np


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
    status: str = "work"


class MineralEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
    ):
        self.render_mode = render_mode

        self.num_trucks = 10
        self.num_places = (3, 3)
        self.capacity = 1  # t
        self.distance = 2*2**0.5  # km
        self.travel_speed = 60 / 60  # km/min
        self.work_speed = 1  # t/min
        self.work_sigma = 0  # t
        self.travel_sigma = 0  # km

        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.num_trucks,
            shape=(2 * self.num_places[0] + self.num_places[1],),
        )
        self.action_space = gym.spaces.Discrete(self.num_places[0])

        if self.render_mode == "human":
            import pygame

            pygame.init()
            self.screen = pygame.display.set_mode((600, 500))
            pygame.font.init()
            self.font = pygame.font.Font(None, 20)
            self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.list_place = [
            [Place() for _ in range(self.num_places[i])] for i in range(2)
        ]
        self.list_truck = [Truck() for _ in range(self.num_trucks)]

        # Initialization
        for i_truck in range(self.num_trucks):
            # truck i travel to shovel i%num_places
            self.list_truck[i_truck].travel_destination = (
                0,
                i_truck % self.num_places[0],
            )
            self.list_truck[i_truck].travel_origin = (
                1,
                i_truck % self.num_places[0],
            )
            self.list_truck[i_truck].travel_remaining = 2 + 0.1 * i_truck
            self.list_truck[i_truck].status = "travel"

        self.worker = self._worker()
        next(self.worker)

        if self.render_mode == "human":
            self.render()

        return "", {}

    def step(self, action: int):
        self.worker.send(action)

        if self.render_mode == "human":
            self.render()

        return None, None, False, False, {}

    def render(self):
        import pygame

        self.screen.fill("black")
        for i in range(2):
            for j in range(self.num_places[i]):
                pos = pygame.Vector2(
                    300 + (2 * i - 1) * 280,
                    50 + 400 * (j + 1) / (self.num_places[i] + 1),
                )
                pygame.draw.circle(self.screen, "dark red", pos, 10)

        for i_truck in range(self.num_trucks):
            truck = self.list_truck[i_truck]
            if truck.status == "travel":
                pos = (
                    truck.travel_remaining
                    * pygame.Vector2(
                        300 + (2 * truck.travel_origin[0] - 1) * 280,
                        50 + 400 * (truck.travel_origin[1] + 1) / (self.num_places[truck.travel_origin[0]] + 1),
                    )
                    + (self.distance - truck.travel_remaining)
                    * pygame.Vector2(
                        300 + (2 * truck.travel_destination[0] - 1) * 280,
                        50 + 400 * (truck.travel_destination[1] + 1) / (self.num_places[truck.travel_destination[0]] + 1),
                    )
                ) / self.distance
                print(f"{pos=}")
                pygame.draw.circle(self.screen, "white", pos, 5)

        pygame.display.flip()
        pygame.event.pump()
        pygame.display.update()

    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()

    def _worker(self):
        while True:
            # Find all remaining times
            list_time_remaining = []
            for i_truck in range(self.num_trucks):
                truck = self.list_truck[i_truck]

                if truck.status == "queue":
                    time_remaining = float("inf")
                else:
                    if truck.status == "work":
                        remaining = truck.work_remaining
                        speed = self.work_speed
                        sigma = self.work_sigma
                    elif truck.status == "travel":
                        remaining = truck.travel_remaining
                        speed = self.travel_speed
                        sigma = self.travel_sigma

                    mean = remaining / speed

                    try:
                        scale = remaining**2 / sigma**2
                    except ZeroDivisionError:
                        time_remaining = mean
                    else:
                        if mean == 0:
                            time_remaining = 0
                        else:
                            time_remaining = self.np_random.wald(mean, scale)

                list_time_remaining.append(time_remaining)

            # Find the truck with the lowest remaining time (truck_first)
            id_truck_first = min(
                range(self.num_trucks), key=list_time_remaining.__getitem__
            )
            remaining_first = list_time_remaining[id_truck_first]
            remaining_first = max(remaining_first, 0)
            print()
            print(f"{list_time_remaining}")
            print(f"{[truck.status for truck in self.list_truck]}")
            print([p.queue for ps in self.list_place for p in ps])
            print([p.current_truck for ps in self.list_place for p in ps])
            print(f"{id_truck_first=} {remaining_first=}")
            truck_first = self.list_truck[id_truck_first]

            # update the others trucks
            for i_truck in range(self.num_trucks):
                if i_truck == id_truck_first:
                    continue

                truck = self.list_truck[i_truck]
                if truck.status == "work":
                    while True:
                        w_t = self.np_random.normal(
                            0, np.sqrt(remaining_first)
                        )
                        displacement = (
                            self.work_speed * remaining_first
                            + self.work_sigma * w_t
                        )
                        if truck.work_remaining >= displacement - 1e-8:
                            truck.work_remaining -= displacement
                            break
                elif truck.status == "travel":
                    while True:
                        w_t = self.np_random.normal(
                            0, np.sqrt(remaining_first)
                        )
                        displacement = (
                            self.travel_speed * remaining_first
                            + self.travel_sigma * w_t
                        )
                        if truck.travel_remaining >= displacement - 1e-8:
                            truck.travel_remaining -= displacement
                            break
                elif truck.status == "queue":
                    pass

            # update the first_truck
            if truck_first.status == "work":
                action = yield

                truck_first.status = "travel"
                truck_first.travel_origin = truck_first.travel_destination
                truck_first.travel_destination = (
                    1 - truck_first.travel_destination[0],
                    action,
                )
                truck_first.travel_remaining = self.distance

                place = self.list_place[truck_first.work_place[0]][
                    truck_first.work_place[1]
                ]
                place.current_truck = None

                if place.queue:
                    i_truck_next = place.queue.pop(0)
                    truck_next = self.list_truck[i_truck_next]
                    truck_next.status = "work"
                    truck_next.work_place = truck_next.travel_destination
                    truck_next.work_remaining = self.capacity
                    place.current_truck = i_truck_next

            elif truck_first.status == "travel":
                place = self.list_place[truck_first.travel_destination[0]][
                    truck_first.travel_destination[1]
                ]
                if place.current_truck is not None:
                    truck_first.status = "queue"
                    place.queue.append(id_truck_first)
                else:
                    truck_first.status = "work"
                    truck_first.work_place = truck_first.travel_destination
                    truck_first.work_remaining = self.capacity
                    place.current_truck = id_truck_first

    def _get_obs(self):
        return (None,), {}
