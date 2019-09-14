import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.summary()
episode_n=[]
mean_score=[]
def update_policy():
 print(len(replay_buffer))
 # exit()
 losses=[]
 losses2=[]
 with tf.GradientTape(persistent=True) as tape:
  for x in replay_buffer:
   for state,action,reward in x:
     logits = model(state)
     # if(action==0):
     #   loss = compute_loss([[1,0]], logits)
     # else:
     #   loss = compute_loss([[0,1]], logits)
     losses.append(-tf.math.log(tf.gather(tf.squeeze(logits),tf.convert_to_tensor(action)))*reward)

  losses=tf.math.reduce_sum(losses)
  losses/=batch_size
  # print(losses2)
  # losses=tf.math.reduce_sum(losses,losses2)
 grads = tape.gradient(losses, model.trainable_variables)
 optimizer.apply_gradients(zip(grads, model.trainable_variables))
 # print(grads)
 # exit()

def discount_normalize_rewards(r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        # print(discounted_r)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r
env = gym.make('CartPole-v0')
env.seed(1)
#env._max_episode_steps = 1000
episodes = 500
batch_size = 10
score=0
replay_buffer=[]
for e in range(episodes):
  
  state = env.reset()
  
  episode_memory = []
  episode_score = 0
  done = False 
  while not done:
    state = state.reshape([1,4])
    logits = model(state)
    # print(logits)
    # print(state)
    # state = state.reshape([1,1,4])
    # print(state)
    # state=[[[ 0.03073904,0.00145001,-0.03088818,-0.03131252]],[[ 0.03073904,0.00145001,-0.03088818,-0.03131252]]]
    # print(np.array(state).shape)
    # print(model.predict(np.array(state),2))
    # exit()
    a_dist = logits.numpy()
    # Choose random action with p = action 
    a = np.random.choice(a_dist[0],p=a_dist[0])
    a, = np.where(a_dist[0] == a)
    a=a[0]
    #need numpy int64
    # if(a==0):
    # 	loss = compute_loss([[1,0]], logits)
    # else:
    # 	loss = compute_loss([[0,1]], logits)
    # make the choosen action 
    next_state, reward, done, _ = env.step(a)
    episode_score +=reward
    episode_memory.append([state,a,reward])
    state=next_state
  episode_memory=np.array(episode_memory)
  episode_memory[:,2] = discount_normalize_rewards(episode_memory[:,2])
  replay_buffer.append(episode_memory)
  score+=episode_score

  print("Episode  {}  Score  {}".format(e+1, episode_score))
  if (e+1) % batch_size == 0:
    episode_n.append(e+1)
    mean_score.append(score/batch_size)
    print("Episode  mean  score  {}".format(score/batch_size))
    update_policy()
    replay_buffer=[]
    score=0
    print("==Policy Updated==")

fig, ax = plt.subplots()
ax.plot(episode_n, mean_score)
ax.set(xlabel='episode n', ylabel='mean score',title=':(')
ax.grid()
fig.savefig("test2.png")
plt.show()
  