import numpy as np
import gym
import sys
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
# actor_model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
#   tf.keras.layers.Dense(4, activation='softmax')
# ])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
inputs = keras.Input(shape=(1,3),)
x = layers.Dense(256, activation='relu')(inputs)
mu = 2*layers.Dense(1, activation='tanh')(x)
sigma = layers.Dense(1, activation='softplus')(x)+ 0.001
actor_model = keras.Model(inputs=inputs, outputs=[mu,sigma], name='actor_model')
actor_model.summary()

e_clip=0.2
ent_coef=0.005

inputs = keras.Input(shape=(1,3),)
x = layers.Dense(256, activation='relu', trainable=False)(inputs)
mu = 2*layers.Dense(1, activation='tanh', trainable=False)(x)
sigma = layers.Dense(1, activation='softplus', trainable=False)(x)+ 0.001
old_actor_model = keras.Model(inputs=inputs, outputs=[mu,sigma], name='old_actor_model')
old_actor_model.summary()

# old_actor_model = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu', trainable=False),
#   tf.keras.layers.Dense(4, activation='softmax', trainable=False)
# ])

inputs = keras.Input(shape=(1,3),)
x = layers.Dense(256, activation='relu')(inputs)
value = layers.Dense(1, activation=None)(x)
critic_model = keras.Model(inputs=inputs, outputs=value, name='critic_model')

critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))
critic_model.summary()

# actor_model.load_weights('./weights_ppo/actor_model1500')
# critic_model.load_weights('./weights_ppo/critic_model1500')
# old_actor_model.load_weights('./weights_ppo/old_actor_model1500')

env = gym.make('Pendulum-v0')
env.seed(1)
env2 = gym.make('Pendulum-v0')
env2.seed(2)
# env._max_episode_steps = 1000
episodes =10000
score=0
episode_n=[]
episode_n_test=[]
score_train=[]
score_test=[]

Actor_update_steps = 10
Critic_update_steps = 10


# @tf.function
def losss(states,actions,advantages):
   # print(states)
   # exit()
   mu, sigma = actor_model(states)
   mu=tf.squeeze(mu)
   sigma=tf.squeeze(sigma)
   normal_dist = tfp.distributions.Normal(mu, sigma)


   entropy_losses = normal_dist.entropy()
   entropy_losses = tf.reduce_mean(entropy_losses, axis=0)

   probs=normal_dist.log_prob(actions)
   
   old_mu, old_sigma = old_actor_model(states)
   old_mu=tf.squeeze(old_mu)
   old_sigma=tf.squeeze(old_sigma)
   old_normal_dist = tfp.distributions.Normal(old_mu, old_sigma)

   old_probs= old_normal_dist.log_prob(actions)
   ratios = tf.exp(probs-old_probs)
   clip_probs = tf.clip_by_value(ratios, 1.-e_clip, 1.+e_clip)


   loss=-tf.reduce_mean(tf.minimum(tf.multiply(ratios, advantages), tf.multiply(clip_probs, advantages)))
   # print(entropy_losses)
   loss=loss-ent_coef*entropy_losses
   return loss

def upd_old_policy():
  weights_actor_model = actor_model.get_weights()
  old_actor_model.set_weights(weights_actor_model)

def train(buff):
    upd_old_policy()
    previous_states= []
    real_previous_values=[]
    advantages=[]
    actions=[]
    last_gae = 0.0
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    for previous_state, action, reward, current_state, done in reversed(buff):
        actions.append(action)
        previous_states.append(previous_state)
        if done:
           delta = reward - critic_model(previous_state)
           last_gae = delta
        else:
          delta = reward + GAMMA * critic_model(current_state) - critic_model(previous_state)
          last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        last_gae=tf.squeeze(last_gae,axis=[1])
        advantages.append(last_gae)
        real_previous_values.append(last_gae + tf.squeeze(critic_model(previous_state),axis=[1]))
 
    actions=list(reversed(actions))
    previous_states=list(reversed(previous_states))
    advantages=list(reversed(advantages))
    real_previous_values=list(reversed(real_previous_values))
    previous_states=tf.squeeze(previous_states,axis=[1])

    previous_states=np.array(previous_states)
    real_previous_values=np.array(real_previous_values)
    
    for _ in range(Critic_update_steps):
      critic_model.train_on_batch(previous_states, real_previous_values)    
  
    advantages=np.vstack(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages=tf.squeeze(advantages)


    for _ in range(Actor_update_steps):
      with tf.GradientTape() as tape:
        losses=losss(previous_states,actions,advantages)

      grads = tape.gradient(losses, actor_model.trainable_variables)
      # grads, _ = tf.clip_by_global_norm(grads, 0.5)
      optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
    
    

def test():
  score=0
  for e in range(20):
    state = env2.reset()
    episode_score = 0
    done = False
    while not done:
       state = state.reshape([1,1,3])
       mu, sigma = actor_model(state)
       normal_dist = tfp.distributions.Normal(mu, sigma)
       action = normal_dist.sample([1,1])
       action=action[0][0][0][0][0]
       action = np.clip(action, -2, 2)

       next_state, reward, done, _ = env2.step([action])
       episode_score +=reward
       state=next_state
    score+=episode_score
  return (score/20)


for e in range(episodes):
  state = env.reset()
  episode_score = 0
  episode_memory=[]
  done = False
  running_add=0
  while not done:
    state = state.reshape([1,1,3])
    mu, sigma = actor_model(state)
    normal_dist = tfp.distributions.Normal(mu, sigma)
    action = normal_dist.sample([1,1])
    action=action[0][0][0][0][0]
    action = np.clip(action, -2, 2)

    next_state, reward, done, _ = env.step([action])
    next_state = next_state.reshape([1,1,3])
    episode_score +=reward

    episode_memory.append([state, action, (reward+8.1)/8.1, next_state, done])
    state=next_state

  episode_memory=np.array(episode_memory)
  train(episode_memory)
  score+=episode_score
  episode_n.append(e+1)
  print("Episode  {}  Score  {}".format(e+1, episode_score))
  score_train.append(episode_score)

  if (e+1) %  10 == 0:
    print("10 Episodes  mean train score {}".format(score/10))
    score=0
  if(e+1) % 500 == 0:
    test_score=test()
    episode_n_test.append(e+1)
    score_test.append(test_score)

    # actor_model.save_weights('./weights_ppo/actor_model'+str(e+1))
    # critic_model.save_weights('./weights_ppo/critic_model'+str(e+1)) 
    # old_actor_model.save_weights('./weights_ppo/old_actor_model'+str(e+1))
    

fig, ax = plt.subplots()
ax.plot(episode_n, score_train)
ax.plot(episode_n_test, score_test)
ax.set(xlabel='episode n', ylabel='score',title=':(')
ax.grid()
fig.legend(['Train score', 'Test score'], loc='upper left')
fig.savefig("ppo_gae_pendulum.png")
plt.show()