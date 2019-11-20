import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
tf.keras.backend.set_floatx('float64')
actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(4, activation='softmax')
])
e_clip=0.2
ent_coef=0.001
# @tf.function
def losss(states,actions,advantages):
   indices=[]
   
   for x in range(len(states)):
     indices.append([x,actions[x]])

   logits=actor_model(states)
   entropy_loss = -tf.reduce_sum(logits *tf.math.log(logits), axis=1)
   entropy_loss = tf.reduce_mean(entropy_loss, axis=0)
   probs=tf.math.log(tf.gather_nd(logits,tf.convert_to_tensor(indices)))
   
   logits2=old_actor_model(states)
   old_probs=tf.math.log(tf.gather_nd(logits2,tf.convert_to_tensor(indices)))
   ratios = tf.exp(probs-old_probs)
   clip_probs = tf.clip_by_value(ratios, 1.-e_clip, 1.+e_clip)

   loss=-tf.reduce_mean(tf.minimum(tf.multiply(ratios, advantages), tf.multiply(clip_probs, advantages)))

   loss=loss-ent_coef*entropy_loss
   return loss


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
old_actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu', trainable=False),
  tf.keras.layers.Dense(4, activation='softmax', trainable=False)
])

def actor_iter(previous_states,actions,advantages):
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
      yield previous_states[ids[i], :],actions[ids[i]],tf.gather(advantages,tf.convert_to_tensor(ids[i]))      

def critic_iter(previous_states,real_previous_values):
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
      yield previous_states[ids[i], :], real_previous_values[ids[i], :]

def upd_old_policy():
  weights_actor_model = actor_model.get_weights()
  old_actor_model.set_weights(weights_actor_model)


critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(1)
])
critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))

# actor_model.load_weights('./weights_ppo_N/actor_model1500')
# critic_model.load_weights('./weights_ppo_N/critic_model1500')
# old_actor_model.load_weights('./weights_ppo_N/old_actor_model1500')

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


Actor_update_steps = 10
Critic_update_steps = 10
batch_size=2048
mini_batch_size=512

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
        advantages.append(last_gae)
        real_previous_values.append(last_gae + critic_model(previous_state))
 
    actions=list(reversed(actions))
    previous_states=list(reversed(previous_states))
    advantages=list(reversed(advantages))
    real_previous_values=list(reversed(real_previous_values))


    previous_states=np.array(previous_states)
    real_previous_values=np.array(real_previous_values)
    for _ in range(Critic_update_steps):
      for prev_states, real_prev_values in critic_iter(previous_states, real_previous_values):
         critic_model.train_on_batch(prev_states, real_prev_values)    

    advantages=np.vstack(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages=tf.squeeze(advantages)
    previous_states=np.vstack(previous_states)
    actions=np.array(actions)
    # print(actions)
    # exit()
    for _ in range(Actor_update_steps):
      for prev_states,acts,advs in actor_iter(previous_states,actions,advantages):
        with tf.GradientTape() as tape:
          losses=losss(prev_states,acts,advs)
        grads = tape.gradient(losses, actor_model.trainable_variables)
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)
        optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
    
    
episode_memory=[]
for e in range(episodes):
  state = env.reset()
  episode_score = 0
  done = False
  while not done:
    state = state.reshape([1,8])
    logits = actor_model(state)
    a_dist = logits.numpy()
    
    a = np.random.choice(a_dist[0],p=a_dist[0]) # Choose random action with p = action 
    a, = np.where(a_dist[0] == a)
    a=a[0]
    next_state, reward, done, _ = env.step(a)
    next_state = next_state.reshape([1,8])
    episode_score +=reward

    episode_memory.append([state, a, reward, next_state, done])
    if (len(episode_memory)==batch_size):
      episode_memory=np.array(episode_memory)
      train(episode_memory)
      print("Policy update")
      episode_memory=[]
    state=next_state
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

    actor_model.save_weights('./weights_ppo_N/actor_model'+str(e+1))
    critic_model.save_weights('./weights_ppo_N/critic_model'+str(e+1)) 
    old_actor_model.save_weights('./weights_ppo_N/old_actor_model'+str(e+1))

fig, ax = plt.subplots()
ax.plot(episode_n, score_train)
ax.plot(episode_n_test, score_test)
ax.set(xlabel='episode n', ylabel='score',title=':(')
ax.grid()
fig.legend(['Train score', 'Test score'], loc='upper left')
fig.savefig("ppo_gae_N.png")
plt.show()