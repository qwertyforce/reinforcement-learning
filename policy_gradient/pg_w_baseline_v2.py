import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])

critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(1)
])
# actor_model.load_weights('./weights_pg/actor_model10000')
# critic_model.load_weights('./weights_pg/critic_model10000')

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
optimizer2 = tf.keras.optimizers.Adam(learning_rate = 0.01)



env = gym.make('CartPole-v0')
env.seed(1)
env2 = gym.make('CartPole-v0')
env2.seed(2)
# env._max_episode_steps = 1000
episodes = 500
batch_size = 10
score=0
episode_n=[]
episode_n_test=[]
score_train=[]
score_test=[]

ent_coef=0.001
replay_buffer=[]

def update_policy():
 losses=[]
 losses2=[]
 entropy_losses=[]
 with tf.GradientTape(persistent=True) as tape:
   for x in replay_buffer:
    states=np.vstack(x[:,0])
    
    actions=x[:,1]
    rewards=x[:,2]
    logits = actor_model(states)

    for v in logits:
      entropy_losses.append(-tf.reduce_sum(v *tf.math.log(v),axis=0))

    values=critic_model(states)
    values=tf.squeeze(values,axis=[1])

    rewards=np.vstack(rewards)
    rewards=tf.convert_to_tensor(rewards, dtype=tf.float64)

    losses2.extend(tf.keras.losses.MSE(rewards,values))

    rewards=rewards-values


    indices=[]

    for x in range(len(states)):
      indices.append([x,0,actions[x]])
    
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

 grads = tape.gradient(losses, actor_model.trainable_variables)
 optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))

 grads2 = tape.gradient(losses2, critic_model.trainable_variables)
 optimizer2.apply_gradients(zip(grads2, critic_model.trainable_variables))


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


def test():
  score=0
  for e in range(20):
    state = env2.reset()
    episode_score = 0
    done = False
    while not done:
       state = state.reshape([1,1,4])
       logits = actor_model(state)
       a_dist = logits.numpy()[0]
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
  
  episode_memory = []
  episode_score = 0
  done = False 
  while not done:
    state = state.reshape([1,1,4])
    logits = actor_model(state)
    a_dist = logits.numpy()[0]

    a = np.random.choice(a_dist[0],p=a_dist[0])  # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0]  #need numpy int64
  
    next_state, reward, done, _ = env.step(a) # make the choosen action 
    episode_score +=reward
    episode_memory.append([state,a,reward])
    state=next_state
  episode_memory=np.array(episode_memory)
  episode_memory[:,2] = discount_normalize_rewards(episode_memory[:,2])
  replay_buffer.append(episode_memory)
  score+=episode_score

  episode_n.append(e+1)
  print("Episode  {}  Score  {}".format(e+1, episode_score))
  score_train.append(episode_score)

  if (e+1) % batch_size == 0:
    update_policy()
    replay_buffer=[]
    print("==Policy Updated==")

  if (e+1) % 10 == 0:
    print("10 episode  mean train score  {}".format(score/10))
    score=0

  if(e+1) % 50== 0:
    test_score=test()
    episode_n_test.append(e+1)
    score_test.append(test_score)

    # actor_model.save_weights('./weights_pg/actor_model'+str(e+1))
    # critic_model.save_weights('./weights_pg/critic_model'+str(e+1))
    

fig, ax = plt.subplots()
ax.plot(episode_n, score_train)
ax.plot(episode_n_test, score_test)
ax.set(xlabel='episode n', ylabel='score',title=':(')
ax.grid()
fig.legend(['Train score', 'Test score'], loc='upper left')
fig.savefig("pg_w_baseline_v2.png")
plt.show()
