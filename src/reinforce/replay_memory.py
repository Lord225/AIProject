from collections import deque
import random
import tensorflow as tf

from reinforce.common import ReplayHistoryType


class ReplayMemory:
    def __init__(self, max_size):
        self.states_buffer = deque(maxlen=max_size)
        self.actions_buffer = deque(maxlen=max_size)
        self.returns_buffer = deque(maxlen=max_size)
        self.next_states_buffer = deque(maxlen=max_size)
        self.dones_buffer = deque(maxlen=max_size)

    def add(self, states, actions, returns, next_states, dones):
        self.states_buffer.extend(states.numpy())
        self.actions_buffer.extend(actions.numpy())
        self.returns_buffer.extend(returns.numpy())
        self.next_states_buffer.extend(next_states.numpy())
        self.dones_buffer.extend(dones.numpy())

    def sample(self, batch_size) -> ReplayHistoryType:
        indices = random.sample(range(len(self.returns_buffer)), batch_size)

        return (
            tf.gather(self.states_buffer, indices),
            tf.gather(self.actions_buffer, indices),
            tf.gather(self.returns_buffer, indices),
            tf.gather(self.next_states_buffer, indices),
            tf.gather(self.dones_buffer, indices)
        )

        # # get random slice of (rnd, rnd+batch_size)
        # rnd = random.randint(0, len(self.returns_buffer) - batch_size)

        # return (
        #     tf.constant(self.states_buffer[rnd:rnd+batch_size]),
        #     tf.constant(self.actions_buffer[rnd:rnd+batch_size]),
        #     tf.constant(self.returns_buffer[rnd:rnd+batch_size]),
        #     tf.constant(self.next_states_buffer[rnd:rnd+batch_size]),
        #     tf.constant(self.dones_buffer[rnd:rnd+batch_size])
        # )
    def __len__(self):
        return len(self.states_buffer)