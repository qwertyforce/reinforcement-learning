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

critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
actor_model.summary()
episode_n=[]
mean_score=[]
ent_coef=0.001
def update_policy():
 # exit()
 losses_actor=[]
 losses_critic=[]
 entropy_losses_actor=[]

 with tf.GradientTape(persistent=True) as tape:
  for x in replay_buffer:
   for state,action,reward in x:
     logits = actor_model(state)
     entropy_losses_actor.append(logits*tf.math.log(logits))
     value=critic_model(state)

     losses_critic.append(tf.keras.losses.MSE(reward,value))
     reward=reward-value
     losses_actor.append(-tf.math.log(tf.gather(tf.squeeze(logits),tf.convert_to_tensor(action)))*reward)
  

  losses_actor=tf.math.reduce_sum(losses_actor)
  losses_critic=tf.math.reduce_sum(losses_critic)
  losses_actor/=batch_size
  losses_critic/=batch_size
  
  entropy_losses_actor=tf.math.reduce_mean(entropy_losses_actor)
  losses_actor=losses_actor-ent_coef*entropy_losses_actor


 grads = tape.gradient(losses_actor, actor_model.trainable_variables)
 # grads, _ = tf.clip_by_global_norm(grads, 0.5)
 optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))

 grads2 = tape.gradient(losses_critic, critic_model.trainable_variables)
 # grads2, _ = tf.clip_by_global_norm(grads2, 0.5)
 optimizer.apply_gradients(zip(grads2, critic_model.trainable_variables))


def discount_normalize_rewards(r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r
env = gym.make('CartPole-v0')
env.seed(1)
# env._max_episode_steps = 1000
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
    logits = actor_model(state)
    a_dist = logits.numpy()
    a = np.random.choice(a_dist[0],p=a_dist[0])  # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0] #need numpy int64

    next_state, reward, done, _ = env.step(a) # make the choosen action 
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
fig.savefig("pg_w_baseline.png")
plt.show()
  