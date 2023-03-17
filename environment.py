from dataclasses import dataclass
from copy import deepcopy
from heapq import heappop
from heapq import heappush
from numpy.random import default_rng
import gymnasium as gym


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

    def _get_obs():
        return tuple([len(q) for q_position in queues for q in q_position] + blocks)

    prev_time = 0
    while True:
        event = heappop(priority_queue)

        #print(event)

        if event.event_type == 0:
            # When truck stop moving and get to the queue

            # If the queue is empty, then trigger the next event for the
            # current truck
            if not queues[event.position][event.index]:
                work_duration = rng.normal(5, 0.1)
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
                #print(f"queue_time=0.000 {event.truck=} {event.position=}")

            # Then, add the truck to the queue
            queues[event.position][event.index].append(event.truck)
            last_time_queue[event.position][event.index] = event.time

        elif event.event_type == 1:
            #  When the truck finish working and start moving.

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
                work_duration = rng.normal(5, 0.1)
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
                #print(f"{queue_time=:.3f} {next_truck=} {event.position=}")

            # Then, choose the next_index and trigger a queue event
            destination = (1 + event.position) % 2
            if event.position == 0:
                next_index = event.index
            else:
                # Obtener obs, rew, done, info
                obs = _get_obs()
                rew = -(event.time - prev_time)
                info = None
                if any(b > 0 for b in blocks):
                    done = False
                else:
                    done = True
                next_index = yield obs, rew, done, False, info

            moving_duration = rng.normal(7, 0.1)
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
    def __init__(self, num_queues, num_trucks, num_blocks):
        self.num_queues = num_queues
        self.num_trucks = num_trucks
        self.num_blocks = num_blocks

        self.observation_space = gym.spaces.Box(
            low=0,
            high=num_trucks, 
            shape=(2 * num_queues[0] + num_queues[1],),
        )
        self.action_space = gym.spaces.Discrete(num_queues[0])

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.w = worker(self.num_queues, self.num_trucks, self.num_blocks, self.np_random)
        obs, rew, done, trunc, info = next(self.w)
        return obs, info

    def step(self, action):
        obs, rew, done, trunc, info = self.w.send(action)
        #print(obs, rew, done, trunc, info)
        return obs, rew, done, trunc, info
