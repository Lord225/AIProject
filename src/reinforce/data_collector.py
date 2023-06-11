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

        action_probs_t = tf.nn.softmax(action_logits_t)

     
        action = tf.cast( 
        tf.squeeze(tf.where(
            tf.random.uniform([1]) < epsilon,
            # Random int, 0-4096
            tf.random.uniform([1], minval=0, maxval=2, dtype=tf.int64),
            # argmax action
            tf.argmax(action_probs_t, axis=1)[0],
        )), dtype=tf.int32)
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


#@tf.function
def run_episode_int_obs(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model, 
        max_steps: int,
        epsilon: float,
        tf_env_step: Callable,
        tf_moves_mask: Callable
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

        mask = tf_moves_mask()

        action_probs_t = tf.nn.softmax(action_logits_t*mask)

        action = tf.cast( 
        tf.squeeze(tf.where(
            tf.random.uniform([1]) < epsilon,
            # Random int, 0-4096
            tf.random.uniform([1], minval=0, maxval=4096, dtype=tf.int64),
            # argmax action
            tf.argmax(action_probs_t, axis=1)[0],
        )), dtype=tf.int32)
        
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

    # Normalize returns
    returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-7)

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

#@tf.function
def run_episode_and_get_history_4(
        initial_state: tf.Tensor,
        actor_model: tf.keras.Model,
        max_steps: int,
        gamma: float,
        epsilon: float,
        tf_env_step: Callable,
        tf_moves_mask: Callable,
) -> Tuple[ReplayHistoryType, tf.Tensor]:
    # run whole episode
    states, action_probs, rewards, next_states, dones = run_episode_int_obs(initial_state, 
                                                                    actor_model, 
                                                                    max_steps, 
                                                                    epsilon,
                                                                    tf_env_step, 
                                                                    tf_moves_mask
                                                                    ) # type: ignore

    # Calculate expected returns
    # returns = get_expected_return(rewards, gamma=gamma) #type: ignore

    return (states, action_probs, rewards, next_states, dones), tf.reduce_sum(rewards)




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


def run_episode_int_obs_selfplay(
        initial_state: tf.Tensor,
        actor_model_white: tf.keras.Model,
        actor_model_black: tf.keras.Model, 
        max_steps: int,
        epsilon_w: float,
        epsilon_b: float,
        tf_env_step: Callable,
        tf_moves_mask: Callable
        ) -> Tuple[ReplayHistoryType, ReplayHistoryType]:
    """
    Run a single episode to collect training data
    collects: 
    * states - state at each step
    * actions - action taken at each step
    * rewards - reward received at each step
    * next_states - next state at each step
    * dones - done flag at each step
    """
    states_white = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions_white = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards_white = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states_white = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones_white = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    states_black = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    actions_black = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    rewards_black = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    next_states_black = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    dones_black = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    state = initial_state

    for t in tf.range(max_steps):
        # white player

        states_white = states_white.write(t, state)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model_white(state) # type: ignore

        mask = tf_moves_mask()

        action_probs_t = tf.nn.softmax(action_logits_t*mask)

        action = tf.cast( 
        tf.squeeze(tf.where(
            tf.random.uniform([1]) < epsilon_w,
            # Random int, 0-4096
            tf.random.uniform([1], minval=0, maxval=4096, dtype=tf.int64),
            # argmax action
            tf.argmax(action_probs_t, axis=1)[0],
        )), dtype=tf.int32)
        
        actions_white = actions_white.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward_white, done_white = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states_white = next_states_white.write(t, state)

        dones_white = dones_white.write(t, tf.cast(done_white, tf.float32))

        if tf.cast(done_white, tf.bool): # type: ignore
            rewards_white = rewards_white.write(t, reward_white)
            break

        # black player
        states_black = states_black.write(t, state)
        state = tf.expand_dims(state, 0)

        # Run the model and to get action probabilities and critic value
        action_logits_t, _ = actor_model_black(state) # type: ignore

        mask = tf_moves_mask()

        action_probs_t = tf.nn.softmax(action_logits_t*mask)

        action = tf.cast(
            tf.squeeze(tf.where(
                tf.random.uniform([1]) < epsilon_b,
                # Random int, 0-4096
                tf.random.uniform([1], minval=0, maxval=4096, dtype=tf.int64),
                # argmax action
                tf.argmax(action_probs_t, axis=1)[0],
            )), dtype=tf.int32)
        
        actions_black = actions_black.write(t, action)

        # Apply action to the environment to get next state and reward
        state, reward_black, done_black = tf_env_step(action)
        state.set_shape(initial_state_shape)

        next_states_black = next_states_black.write(t, state)

        dones_black = dones_black.write(t, tf.cast(done_black, tf.float32))

        rewards_white = rewards_white.write(t, reward_white - 0.9*reward_black)
        rewards_black = rewards_black.write(t, reward_black - 0.9*reward_white)

        if tf.cast(done_black, tf.bool): # type: ignore
            break


    states_white = states_white.stack()
    actions_white = actions_white.stack()
    rewards_white = rewards_white.stack()
    next_states_white = next_states_white.stack()
    dones_white = dones_white.stack()

    states_black = states_black.stack()
    actions_black = actions_black.stack()
    rewards_black = rewards_black.stack()
    next_states_black = next_states_black.stack()
    dones_black = dones_black.stack()

    return (states_white, actions_white, rewards_white, next_states_white, dones_white), (states_black, actions_black, rewards_black, next_states_black, dones_black)



    
def run_episode_and_get_history_selfplay(
        initial_state: tf.Tensor,
        actor_model_white: tf.keras.Model,
        actor_model_black: tf.keras.Model,
        max_steps: int,
        epsilon_w: float,
        epsilon_b: float,
        tf_env_step: Callable,
        tf_moves_mask: Callable,
) -> Tuple[ReplayHistoryType, ReplayHistoryType, tf.Tensor, tf.Tensor]:
    # run whole episode
    white, black = run_episode_int_obs_selfplay(initial_state, 
                                                actor_model_white, 
                                                actor_model_black, 
                                                max_steps, 
                                                epsilon_w,
                                                epsilon_b, 
                                                tf_env_step, 
                                                tf_moves_mask)

    # Calculate expected returns
    # returns = get_expected_return(rewards, gamma=gamma) #type: ignore
    rewards_white = white[2]
    rewards_black = black[2]

    return white, black, tf.reduce_sum(rewards_white), tf.reduce_sum(rewards_black)