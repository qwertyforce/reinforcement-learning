import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt

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
episodes = 1000
score=0
episode_n=[]
mean_score=[]
discount_factor=0.99
max_score=200

def train(previous_state, action, reward, current_state, done):
    previous_state_predicted_value=critic_model(previous_state)

    if not done:
        current_state_predicted_value=critic_model(current_state)
    else:
        current_state_predicted_value=0

    real_previous_value=np.zeros((1,1,1))
    real_previous_value = reward + discount_factor * current_state_predicted_value
    advantages = np.zeros((1,1,2))
    advantages[0][0][action] = real_previous_value - previous_state_predicted_value
    previous_state=previous_state.reshape([1,1,4])
    

    real_previous_value=tf.reshape(real_previous_value,[1,1,1])
    actor_model.fit(previous_state, advantages, epochs=1, verbose=0)
    critic_model.fit(previous_state, real_previous_value, epochs=1, verbose=0)
    
    
	
replay_buffer=[]
for e in range(episodes):
  state = env.reset()
  episode_score = 0
  done = False 
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
    	reward=-30
    train(state, a, reward, next_state, done)
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
fig.savefig("test.png")
plt.show()