from typing import Callable, List, Tuple
import tensorflow as tf

from reinforce.common import ReplayHistoryType

loss_fn = tf.keras.losses.mean_squared_error

@tf.function
def training_step_dqnet_target_critic(
        batch: ReplayHistoryType,
        minibatch_size: int,
        train_iterations: int,
        discount_rate: float,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        n_outputs: int
):
    """
    Training step that uses:
    - DQN target network
    - Critic network & cost
    - Double DQN
    - Mulitple minibatch iterations
    """
    # sample minibatch_size experiences from batch
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

    for _ in tf.range(train_iterations):
        idxs = tf.random.uniform(shape=(minibatch_size,), minval=0, maxval=len(batch_states), dtype=tf.int32)

        states = tf.gather(batch_states, idxs)
        actions = tf.gather(batch_actions, idxs)
        rewards = tf.gather(batch_rewards, idxs)
        next_states = tf.gather(batch_next_states, idxs)
        dones = tf.gather(batch_dones, idxs)

        # next_Q_values, _ = target_model(next_states, training=True) # type: ignore
        # max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        # target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
        # target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        # mask = tf.one_hot(actions, n_outputs)

        next_Q_values, _ = actor_model(next_states, training=True) # type: ignore
        best_next_actions = tf.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, n_outputs)
        next_best_Q, _ = target_model(next_states, training=True) # type: ignore 
        next_best_Q_values = tf.reduce_sum(next_best_Q * next_mask, axis=1)
        target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        mask = tf.one_hot(actions, n_outputs)

        with tf.GradientTape() as tape:
            all_Q_values, values = actor_model(states, training=True) # type: ignore
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, actor_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))

@tf.function
def training_step_dqnet_target_critic_custom_action(
        batch: ReplayHistoryType,
        minibatch_size: int,
        train_iterations: int,
        discount_rate: float,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        tf_transform_state: Callable,
        tf_transform_action: Callable,
):
    """
    Training step that uses:
    - DQN target network
    - Critic network & cost
    - Double DQN
    - Mulitple minibatch iterations
    """
    # sample minibatch_size experiences from batch
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

    for _ in tf.range(train_iterations):
        idxs = tf.random.uniform(shape=(minibatch_size,), minval=0, maxval=len(batch_states), dtype=tf.int32)

        states = tf.gather(batch_states, idxs)
        actions = tf.gather(batch_actions, idxs)
        rewards = tf.gather(batch_rewards, idxs)
        next_states = tf.gather(batch_next_states, idxs)
        dones = tf.gather(batch_dones, idxs)

        # next_Q_values, _ = target_model(next_states, training=True) # type: ignore
        # max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
        # target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
        # target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        # mask = tf.one_hot(actions, n_outputs)

        states_transformed = tf_transform_state(states)
        next_Q_values, _ = actor_model(states_transformed, training=True) # type: ignore
        best_next_actions = tf_transform_action(next_states, next_Q_values)[0]
        
        #next_mask = tf.one_hot(best_next_actions, n_outputs)
        nmask1 = tf.reshape(tf.one_hot(best_next_actions% 64, 64), (-1, 8, 8, 1))
        nmask2 = tf.reshape(tf.one_hot(best_next_actions//64, 64), (-1, 8, 8, 1))
        next_mask = tf.concat([nmask1, nmask2], axis=3)

        next_states_transformed = tf_transform_state(states)
        next_best_Q, _ = target_model(next_states_transformed, training=True) # type: ignore 
        next_best_Q_values =  tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(next_best_Q * next_mask, axis=1), axis=1), axis=1)
        target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * next_best_Q_values)
        target_Q_values = tf.reshape(target_Q_values, [-1, 1])
        
        # mask for this action space needs to create two one-hot vectors (8x8) for each action 
        # mask = tf.concat(tf.reshape(tf.one_hot(actions%64, 64), (8, 8)), tf.reshape(tf.one_hot(actions//64, 64), (8, 8)), axis=1) # type: ignore
        mask1 = tf.reshape(tf.one_hot(actions% 64, 64), (-1, 8, 8, 1))
        mask2 = tf.reshape(tf.one_hot(actions//64, 64), (-1, 8, 8, 1))
        mask = tf.concat([mask1, mask2], axis=3)

        with tf.GradientTape() as tape:
            all_Q_values, values = actor_model(next_states_transformed, training=True) # type: ignore
            # Q_values = tf.reduce_sum(all_Q_values * mask, axis=1)
            Q_values = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(all_Q_values * mask, axis=1), axis=1), axis=1, keepdims=True)

            actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
            critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

            loss = actor_loss + critic_loss
        grads = tape.gradient(loss, actor_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))



@tf.function
def training_step_target_critic(
        batch: ReplayHistoryType,
        target_model: tf.keras.Model,
        actor_model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        n_outputs,
        discount_rate
):
    """
    Training step that uses:
    - DQN target network
    - Critic network & cost
    """
    states, actions, rewards, next_states, dones = batch
    next_Q_values, _ = target_model(next_states, training=True) # type: ignore
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values, values = actor_model(states, training=True) # type: ignore
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        actor_loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
        critic_loss = tf.reduce_mean(loss_fn(target_Q_values, values))

        loss = actor_loss + critic_loss
    grads = tape.gradient(loss, actor_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))


@tf.function
def training_step_no_critic_no_target(
        batch: ReplayHistoryType,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        n_outputs: int,
        discount_rate: float
):
    """
    Baseline training step that does not use any fancy stuff
    """
    # states, actions, rewards, next_states, dones = batch
    # next_Q_values = model.predict(next_states, verbose=0)
    # max_next_Q_values = np.max(next_Q_values, axis=1)
    # target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
    # target_Q_values = target_Q_values.reshape(-1, 1)
    # mask = tf.one_hot(actions, n_outputs)
    # with tf.GradientTape() as tape:
    #     all_Q_values = model(states)
    #     Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
    #     loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    # grads = tape.gradient(loss, model.trainable_variables)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # tf function version
    states, actions, rewards, next_states, dones = batch
    next_Q_values = model(next_states, training=True)
    max_next_Q_values = tf.reduce_max(next_Q_values, axis=1)
    target_Q_values = (rewards + (tf.constant(1.0, dtype=tf.float32) - dones) * discount_rate * max_next_Q_values)
    target_Q_values = tf.reshape(target_Q_values, [-1, 1])
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states, training=True)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
