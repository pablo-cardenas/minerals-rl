from copy import deepcopy
from dataclasses import dataclass
from heapq import heappop
from heapq import heappush
from typing import Optional
from itertools import groupby

import gymnasium as gym
from numpy.random import default_rng


@dataclass(frozen=True, order=True)
class Event:
    """
    Event
    -----

    time: time of the event
    truck: index of the truck
    type:
      - 0 when the truck stop moving and get to the queue.
      - 1 when the truck finish working and start moving.
    position:  the current position of the truck
      - if type = 0 : The place of the queue
      - if type = 1 : the place where the truck just finished working
    index: index of the position
    """

    time: int
    truck: int
    event_type: int
    position: int
    index: int

    def __repr__(self):
        return (
            "Event("
            f"time={self.time:.3f}, "
            f"truck={self.truck}, "
            f"event_type={self.event_type}, "
            f"position={self.position}, "
            f"index={self.index})"
        )


def worker(num_queues, num_trucks, num_blocks, rng):
    # Initialize queues
    blocks = [num_blocks] * num_queues[1]
    queue_times = []
    last_time_queue = [
        [0] * num_queues[0],
        [0] * num_queues[1],
    ]
    queues = [
        [[] for _ in range(num_queues[0])],
        [[] for _ in range(num_queues[1])],
    ]
    going = [
        [[] for _ in range(num_queues[0])],
        [[] for _ in range(num_queues[1])],
    ]
    priority_queue = []
    for i in range(num_trucks):
        e = Event(
            time=0,
            truck=i,
            event_type=0,
            position=0,
            index=i % num_queues[0],
        )
        heappush(priority_queue, e)

    prev_time = 0
    while True:
        event = heappop(priority_queue)

        # print(event)

        if event.event_type == 0:
            # When truck stop moving and get to the queue

            # If the queue is empty, then trigger the next event for the
            # current truck
            if not queues[event.position][event.index]:
                work_duration = rng.normal(5, 0)
                new_event = Event(
                    time=event.time + work_duration,
                    truck=event.truck,
                    event_type=1,
                    position=event.position,
                    index=event.index,
                )
                heappush(priority_queue, new_event)

                # Store the queue time
                queue_times.append(0)
                # print(f"queue_time=0.000 {event.truck=} {event.position=}")

            # Then, add the truck to the queue
            queues[event.position][event.index].append(event.truck)
            last_time_queue[event.position][event.index] = event.time

        elif event.event_type == 1:
            # When the truck finish working and start moving.

            # Remove the truck from the queue
            assert queues[event.position][event.index]
            queues[event.position][event.index].pop(0)
            # Discount blocks
            if event.position == 1:
                blocks[event.index] -= 1

            if queues[event.position][event.index]:
                # If the queue is not empty, then trigger the next event for
                # the next truck in the queue
                next_truck = queues[event.position][event.index][0]
                work_duration = rng.normal(5, 0)
                new_event = Event(
                    time=event.time + work_duration,
                    truck=next_truck,
                    event_type=1,
                    position=event.position,
                    index=event.index,
                )
                heappush(priority_queue, new_event)

                # Store the queue time
                queue_time = (
                    event.time - last_time_queue[event.position][event.index]
                )
                queue_times.append(queue_time)
                # print(f"{queue_time=:.3f} {next_truck=} {event.position=}")

            # Then, choose the next_index and trigger a queue event
            destination = (1 + event.position) % 2
            print(event)
            if event.position == 0:
                next_index = event.index
                obs = deepcopy(priority_queue), deepcopy(queues)
                next_index = yield obs, 0, False, False, {}
            else:
                # Obtener obs, rew, done, info
                obs = deepcopy(priority_queue), deepcopy(queues)
                rew = -(event.time - prev_time)
                if any(b > 0 for b in blocks):
                    done = False
                else:
                    done = True
                next_index = yield obs, rew, done, False, {}

            moving_duration = rng.normal(7, 0)
            new_event = Event(
                time=event.time + moving_duration,
                truck=event.truck,
                event_type=0,
                position=destination,
                index=next_index,
            )
            heappush(priority_queue, new_event)

        # End While and upadte the prev_time
        prev_time = event.time


class MineralEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_queues,
        num_trucks,
        num_blocks,
        render_mode: Optional[str] = None,
        natural=False,
        sab=False,
    ):
        self.render_mode = render_mode
        self.num_queues = num_queues
        self.num_trucks = num_trucks
        self.num_blocks = num_blocks

        self.observation_space = gym.spaces.Box(
            low=0,
            high=num_trucks,
            shape=(2 * num_queues[0] + num_queues[1],),
        )
        self.action_space = gym.spaces.Discrete(num_queues[0])

        try:
            import pygame

        except ImportError as e:
            raise DependencyNotInstalled("pygame is not installed") from e

        pygame.init()
        screen_width, screen_height = 600, 500
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.w = worker(
            self.num_queues, self.num_trucks, self.num_blocks, self.np_random
        )
        state, rew, done, trunc, info = next(self.w)
        self.state = state

        if self.render_mode == "human":
            self.render()

        return state, info

    def step(self, action):
        state, rew, done, trunc, info = self.w.send(action)
        self.state = state

        # print(obs, rew, done, trunc, info)

        if self.render_mode == "human":
            self.render()

        return state, rew, done, trunc, info

    def render(self):
        import pygame
        priority_queue, queues = self.state

        self.screen.fill('black')
        for i in range(2):
            for j in range(self.num_queues[i]):
                pos = pygame.Vector2(300 + (2*i-1) * 280, 50 + 400 * (j + 1)/(self.num_queues[0] + 1))
                pygame.draw.circle(self.screen, "dark red", pos, 10)
                text = self.font.render(str(queues[i][j]), True, 'white')
                self.screen.blit(text, pos + (-20*(2*i-1) - i*text.get_rect().right, -5))



        for event in priority_queue:
            if event.event_type == 1:
                i = event.position
                j = event.index
                pos = pygame.Vector2(300 + (2*i-1) * 280, 50 + 400 * (j+1)/(self.num_queues[i] + 1))
                text = self.font.render("{:.1f}".format(event.time), True, 'white')
                self.screen.blit(text, pos + (+10 *(2*i-1) - i*text.get_rect().right, -5))

        for i in range(2):
            for j in range(self.num_queues[0]):
                pos = pygame.Vector2(300 + (2*i-1) * 280, 50 + 400 * (j+1)/(self.num_queues[0] + 1))
                L = []
                for event in priority_queue:
                    if event.event_type == 0 and event.position == i and event.index == j:
                        L.append(f"({event.time:.1f}, {event.truck})")
                    
                text = self.font.render("[" + ",".join(L) + "]", True, 'white')
                self.screen.blit(text, pos + (-90 *(2*i-1) - i * text.get_rect().right, -5))
                
            
        pygame.display.flip()


    def close(self):
        if hasattr(self, "screen"):
            import pygame

            pygame.display.quit()
            pygame.quit()
