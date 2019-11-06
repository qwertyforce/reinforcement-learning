import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
])

model2 = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
optimizer2 = tf.keras.optimizers.Adam(learning_rate = 0.01)
model.summary()
episode_n=[]
mean_score=[]
ent_coef=0.001
def update_policy():
 losses=[]
 losses2=[]
 entropy_losses=[]
 with tf.GradientTape(persistent=True) as tape:
   for x in replay_buffer:
    states=np.vstack(x[:,0])
    
    actions=x[:,1]
    rewards=x[:,2]

    logits = model(states)
    for v in logits:
      entropy_losses.append(-tf.reduce_sum(v *tf.math.log(v),axis=0))

    values=model2(states)
   
    rewards=np.vstack(rewards)
    rewards=tf.convert_to_tensor(rewards, dtype=tf.float64)

    losses2.extend(tf.keras.losses.MSE(rewards,values))
    
    rewards=rewards-values
    indices=[]
    for x in range(len(states)):
      indices.append([x,actions[x]])

    rewards=tf.squeeze(rewards)

    neg_log_prob=-tf.math.log(tf.gather_nd(logits,tf.convert_to_tensor(indices)))
    losses.extend(tf.math.multiply(neg_log_prob,tf.convert_to_tensor(rewards)))

   losses=tf.math.reduce_sum(losses)
   losses2=tf.math.reduce_sum(losses2)
   losses/=batch_size
   losses2/=batch_size

   entropy_losses = tf.reduce_mean(entropy_losses)
   # print(entropy_losses)
   losses=losses-ent_coef*entropy_losses

 grads = tape.gradient(losses, model.trainable_variables)
 optimizer.apply_gradients(zip(grads, model.trainable_variables))

 grads2 = tape.gradient(losses2, model2.trainable_variables)
 optimizer2.apply_gradients(zip(grads2, model2.trainable_variables))


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
env = gym.make('LunarLander-v2')
env.seed(1)
# env._max_episode_steps = 1000
episodes = 10000
batch_size = 10
score=0
replay_buffer=[]
for e in range(episodes):
  
  state = env.reset()
  
  episode_memory = []
  episode_score = 0
  done = False 
  while not done:
    state = state.reshape([1,8])
    logits = model(state)
    a_dist = logits.numpy()
    
    a = np.random.choice(a_dist[0],p=a_dist[0])  # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0]  #need numpy int64
  
    next_state, reward, done, _ = env.step(a) # make the choosen action 
    episode_score +=reward
    episode_memory.append([state.reshape(8,),a,reward])
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
fig.savefig("pg_w_baseline_v2.png")
plt.show()
  