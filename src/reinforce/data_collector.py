from typing import Callable, List, Tuple
import tensorflow as tf

from reinforce.common import ReplayHistoryType

@tf.function
def run_episode(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int,
        epsilon: float,
        tf_env_step: Callable
        ) -> ReplayHistoryType:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        states = states.write(t, state)

        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model(state) # type: ignore

        if tf.random.uniform(()) < epsilon:
            # take random action
            action = tf.random.uniform((1,), minval=0, maxval=4096, dtype=tf.int32)[0]
        else:
            # Sample next action from the action probability distribution        
            action = tf.random.categorical(tf.nn.softmax(action_logits_t), 1, dtype=tf.int32)[0, 0]
        
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states = next_states.write(t, state)

        dones = dones.write(t, tf.cast(done, tf.float32))
        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()
    dones = dones.stack()

    return states, actions, rewards, next_states, dones


@tf.function
def run_episode_int_obs(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int,
        epsilon: float,
        tf_env_step: Callable
        ) -> ReplayHistoryType:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        states = states.write(t, state)

        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model(state) # type: ignore

        if tf.random.uniform(()) < epsilon:
            # take random action
            action = tf.random.uniform((1,), minval=0, maxval=4096, dtype=tf.int32)[0]
        else:
            # Sample next action from the action probability distribution        
            action = tf.random.categorical(tf.nn.softmax(action_logits_t), 1, dtype=tf.int32)[0, 0]
        
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states = next_states.write(t, state)

        dones = dones.write(t, tf.cast(done, tf.float32))
        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()
    dones = dones.stack()

    return states, actions, rewards, next_states, dones

@tf.function
def run_episode_observation_transform(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int,
        tf_env_step: Callable,
        tf_transform_state: Callable,
        epsilon: float
        ) -> ReplayHistoryType:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        states = states.write(t, state)

        #state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        state_transformed = tf_transform_state(state)
        action_logits_t, _ = actor_model(tf.reshape(state_transformed, (1, 8, 8, 8))) # type: ignore


        # epsilon greedy policy
        if tf.random.uniform(()) < epsilon:
            # take random action
            action = tf.random.uniform((1,), minval=0, maxval=4096, dtype=tf.int32)[0]
        else:
            # Sample next action from the action probability distribution        
            action = tf.random.categorical(tf.nn.softmax(action_logits_t), 1, dtype=tf.int32)[0, 0]
        
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states = next_states.write(t, state)

        dones = dones.write(t, tf.cast(done, tf.float32))
        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()
    dones = dones.stack()

    return states, actions, rewards, next_states, dones

@tf.function
def run_episode_custom_action(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int,
        tf_env_step: Callable,
        tf_transform_action: Callable,
        tf_transform_state: Callable
        ) -> ReplayHistoryType:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states = tf.TensorArray(dtype=tf.int8, size=0, dynamic_size=True)
    dones = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # Convert state into a batched tensor (batch size = 1)
        states = states.write(t, state)

        state = tf_transform_state(state)
        #state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model(tf.reshape(state, (1, 8, 8, 8))) # type: ignore

        # Sample next action from the action probability distribution
        #action = tf.random.categorical(action_logits_t, 1, dtype=tf.int32)[0, 0]
        action = tf_transform_action(action_logits_t)
        actions = actions.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward, done = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states = next_states.write(t, state)

        dones = dones.write(t, tf.cast(done, tf.float32))
        # Store reward
        rewards = rewards.write(t, reward)

        if tf.cast(done, tf.bool):
            break

    states = states.stack()
    actions = actions.stack()
    rewards = rewards.stack()
    next_states = next_states.stack()
    dones = dones.stack()

    return states, actions, rewards, next_states, dones

def get_expected_return(
    rewards: List, 
    gamma: float):
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32) # type: ignore
    discounted_sum = tf.constant(0.0)
    discounted_sum_shape = discounted_sum.shape
    for i in tf.range(n):
        reward = rewards[i]
        discounted_sum = reward + gamma * discounted_sum
        discounted_sum.set_shape(discounted_sum_shape)
        returns = returns.write(i, discounted_sum)

    returns = returns.stack()[::-1] # type: ignore

    return returns

@tf.function
def run_episode_and_get_history(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float,
        epsilon: float,
        tf_env_step: Callable
) -> Tuple[ReplayHistoryType, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode(initial_state, actor_model, max_steps, epsilon, tf_env_step) # type: ignore

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, returns, next_states, dones), tf.reduce_sum(rewards)

@tf.function
def run_episode_and_get_history_3(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float,
        epsilon: float,
        tf_env_step: Callable,
        tf_transform_state: Callable,
) -> Tuple[ReplayHistoryType, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode_observation_transform(initial_state, 
                                                                    actor_model, 
                                                                    max_steps, 
                                                                    tf_env_step,
                                                                    tf_transform_state,
                                                                    epsilon
                                                                    ) # type: ignore

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, returns, next_states, dones), tf.reduce_sum(rewards)

@tf.function
def run_episode_and_get_history_4(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float,
        epsilon: float,
        tf_env_step: Callable,
) -> Tuple[ReplayHistoryType, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode_int_obs(initial_state, 
                                                                    actor_model, 
                                                                    max_steps, 
                                                                    epsilon,
                                                                    tf_env_step
                                                                    ) # type: ignore

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, returns, next_states, dones), tf.reduce_sum(rewards)


@tf.function
def run_episode_and_get_history_2(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float,
        tf_env_step: Callable,
        tf_transform_action: Callable,
        tf_transform_state: Callable
) -> Tuple[ReplayHistoryType, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode_custom_action(initial_state, 
                                                                    actor_model, 
                                                                    max_steps, 
                                                                    tf_env_step,
                                                                    tf_transform_action,
                                                                    tf_transform_state
                                                                    ) # type: ignore

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, returns, next_states, dones), tf.reduce_sum(rewards)