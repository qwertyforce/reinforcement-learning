import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')
actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
])
actor_model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001))

critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(1)
])
critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))


# actor_model.load_weights('./weights_a2c/actor_model5000')
# critic_model.load_weights('./weights_a2c/critic_model5000')
 

env = gym.make('LunarLander-v2')
env.seed(1)
env2 = gym.make('LunarLander-v2')
env2.seed(2)
# env._max_episode_steps = 1000
episodes =10000
score=0
episode_n=[]
episode_n_test=[]
score_train=[]
score_test=[]

def train2(previous_states,advantages,real_previous_values):
    actor_model.train_on_batch(previous_states, advantages)
    critic_model.train_on_batch(previous_states, real_previous_values)
    
def train(buff):
    previous_states= []
    real_previous_values=[]
    advantages=[]

    last_gae = 0.0
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    for previous_state, action, reward, current_state, done in reversed(buff):
        previous_states.append(previous_state)
        if done:
           delta = reward - critic_model(previous_state)
           last_gae = delta
        else:
          delta = reward + GAMMA * critic_model(current_state) - critic_model(previous_state)
          last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        advantage=np.zeros((1,4))
        advantage[0][action]=last_gae
        advantages.append(advantage)
        real_previous_values.append(last_gae + critic_model(previous_state))
 
    previous_states=list(reversed(previous_states))
    advantages=list(reversed(advantages))
    real_previous_values=list(reversed(real_previous_values))
    
    previous_states=tf.convert_to_tensor(previous_states)

    real_previous_values=tf.convert_to_tensor(real_previous_values)
    advantages=tf.convert_to_tensor(advantages)

    train2(previous_states,advantages,real_previous_values)

def test():
  score=0
  for e in range(20):
    state = env2.reset()
    episode_score = 0
    done = False
    while not done:
       state = state.reshape([1,8])
       logits = actor_model(state)
       a_dist = logits.numpy()
       a = np.random.choice(a_dist[0],p=a_dist[0]) # Choose random action with p = action 
       a, = np.where(a_dist[0] == a)
       a=a[0]
       next_state, reward, done, _ = env2.step(a)
       episode_score +=reward
       state=next_state
    score+=episode_score
  return (score/20)

for e in range(episodes):
  state = env.reset()
  episode_score = 0
  episode_memory=[]
  done = False
  replay_buffer=[]
  running_add=0
  while not done:
    state = state.reshape([1,8])
    logits = actor_model(state)
    a_dist = logits.numpy()
    # env.render()
    a = np.random.choice(a_dist[0],p=a_dist[0]) # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0]
    next_state, reward, done, _ = env.step(a)
    next_state = next_state.reshape([1,8])
    episode_score +=reward

    # if done and not(episode_score==max_score):
    # 	running_add=-30

    episode_memory.append([state, a, reward, next_state, done])
    state=next_state
  episode_memory=np.array(episode_memory)
  train(episode_memory)
  score+=episode_score
  episode_n.append(e+1)
  score_train.append(episode_score)
  print("Episode  {}  Score  {}".format(e+1, episode_score))
  if (e+1) % 10 == 0:
    print("10 Episodes  mean train score  {}".format(score/10))
    score=0

  if(e+1) % 500 == 0:
    test_score=test()
    episode_n_test.append(e+1)
    score_test.append(test_score)

    actor_model.save_weights('./weights_a2c/actor_model'+str(e+1))
    critic_model.save_weights('./weights_a2c/critic_model'+str(e+1))
    

fig, ax = plt.subplots()
ax.plot(episode_n, score_train)
ax.plot(episode_n_test, score_test)
ax.set(xlabel='episode n', ylabel='score',title=':(')
ax.grid()
fig.legend(['Train score', 'Test score'], loc='upper left')
fig.savefig("a2c_MC_gae.png")
plt.show()