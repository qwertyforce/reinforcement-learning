import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
tf.keras.backend.set_floatx('float64')

actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])
actor_model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001))

critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(1)
])
critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))


env = gym.make('CartPole-v0')
env.seed(1)
episodes = 500
score=0
episode_n=[]
mean_score=[]
discount_factor=0.99
max_score=200
batch_size=64

def train2(previous_states,advantages,real_previous_values,buff_size):
    actor_model.fit(previous_states, advantages, epochs=1, verbose=0,batch_size=buff_size)
    critic_model.fit(previous_states, real_previous_values, epochs=1,verbose=0,batch_size=buff_size)
    
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
        advantage=np.zeros((1,2))
        advantage[0][action]=last_gae
        advantages.append(advantage)
        real_previous_values.append(last_gae + critic_model(previous_state))
 
    previous_states=list(reversed(previous_states))
    advantages=list(reversed(advantages))
    real_previous_values=list(reversed(real_previous_values))
    
    previous_states=tf.convert_to_tensor(previous_states)

    real_previous_values=tf.convert_to_tensor(real_previous_values)
    advantages=tf.convert_to_tensor(advantages)
    batch_size=tf.Variable(float(len(buff)))

    train2(previous_states,advantages,real_previous_values,batch_size)


    
    
episode_memory=[]
dones=[]
for e in range(episodes):
  state = env.reset()
  episode_score = 0

  done = False

  while not done:
    state = state.reshape([1,4])
    logits = actor_model(state)
    a_dist = logits.numpy()
    
    a = np.random.choice(a_dist[0],p=a_dist[0]) # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0]
    next_state, reward, done, _ = env.step(a)
    next_state = next_state.reshape([1,4])
    episode_score +=reward
    dones.append(done)

    # if done and not(episode_score==max_score):
    # 	running_add=-30

    episode_memory.append([state, a, reward, next_state, done])
    
    if (len(episode_memory)==batch_size):
      episode_memory=np.array(episode_memory)
      train(episode_memory)
      print("Policy update")
      episode_memory=[]
      dones=[]
    state=next_state
  score+=episode_score

  print("Episode  {}  Score  {}".format(e+1, episode_score))
  if (e+1) % 10 == 0:
    episode_n.append(e+1)
    mean_score.append(score/10)
    print("Episode  mean  score  {}".format(score/10))
    score=0
    

fig, ax = plt.subplots()
ax.plot(episode_n, mean_score)
ax.set(xlabel='episode n', ylabel='score',title=':(')
ax.grid()
fig.savefig("a2c_GAE_N.png")
plt.show()