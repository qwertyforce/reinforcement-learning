import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
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

def discount_normalize_rewards(running_add,r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    # running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        # print(discounted_r)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)+1e-8
    return discounted_r

    
    # if(np.std(discounted_r)==0):
    #   print(r.zise)
    #   print(discounted_r)
    
    

env = gym.make('CartPole-v0')
env.seed(1)
episodes = 500
score=0
episode_n=[]
mean_score=[]
discount_factor=0.99
max_score=200
batch_size=64
def train(buff):
    previous_states= []
    real_previous_values=[]
    advantages=[]

    for previous_state, action, reward, current_state, done in buff:
        previous_states.append(previous_state)
        
        if not done:
          current_state_predicted_value=critic_model(current_state)
        else:
          current_state_predicted_value=0

        previous_state_predicted_value=critic_model(previous_state)
        real_previous_value = reward + discount_factor * current_state_predicted_value
        real_previous_values.append(real_previous_value)
        advantage=np.zeros((1,2))
        advantage[0][action]=real_previous_value - previous_state_predicted_value
        advantages.append(advantage)

    
    previous_states=np.array(previous_states)
    real_previous_values=np.array(real_previous_values)
    advantages=np.array(advantages)
    
    # actor_model.train_on_batch(previous_states, advantages)
    # critic_model.train_on_batch(previous_states, real_previous_values)
    actor_model.fit(previous_states, advantages, epochs=1, verbose=0,batch_size=len(buff))
    critic_model.fit(previous_states, real_previous_values, epochs=1,verbose=0,batch_size=len(buff))
    
    
	
for e in range(episodes):
  state = env.reset()
  episode_score = 0
  episode_memory=[]
  done = False
  replay_buffer=[]
  running_add=0
  while not done:
    state = state.reshape([1,4])
    logits = actor_model(state)
    a_dist = logits.numpy()
    # Choose random action with p = action 
    a = np.random.choice(a_dist[0],p=a_dist[0])
    a, = np.where(a_dist[0] == a)
    a=a[0]
    next_state, reward, done, _ = env.step(a)
    next_state = next_state.reshape([1,4])
    episode_score +=reward

    if done and not(episode_score==max_score):
    	running_add=-30

    episode_memory.append([state, a, reward, next_state,done])
    if (len(episode_memory)==batch_size or done):
      # running_add=critic_model(next_state)
      # running_add=running_add.numpy()[0][0]
      episode_memory=np.array(episode_memory)
      episode_memory[:,2] = discount_normalize_rewards(running_add,episode_memory[:,2])
      train(episode_memory)
      episode_memory=[]
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
fig.savefig("aw.png")
plt.show()