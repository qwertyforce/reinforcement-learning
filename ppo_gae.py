import numpy as np
import gym
import sys
import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp
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

   entropy_losses = -tf.reduce_sum(logits *tf.math.log(logits), axis=1)
   entropy_losses = tf.reduce_mean(entropy_losses, axis=0)

   probs=tf.math.log(tf.gather_nd(logits,tf.convert_to_tensor(indices)))
   
   logits2=old_actor_model(states)
   old_probs=tf.math.log(tf.gather_nd(logits2,tf.convert_to_tensor(indices)))
   ratios = tf.exp(probs-old_probs)
   clip_probs = tf.clip_by_value(ratios, 1.-e_clip, 1.+e_clip)


   loss=-tf.reduce_mean(tf.minimum(tf.multiply(ratios, advantages), tf.multiply(clip_probs, advantages)))
   # print(entropy_losses)
   loss=loss-ent_coef*entropy_losses
   return loss


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
old_actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu', trainable=False),
  tf.keras.layers.Dense(4, activation='softmax', trainable=False)
])


def upd_old_policy():
  weights_actor_model = actor_model.get_weights()
  old_actor_model.set_weights(weights_actor_model)


critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,8),activation='relu'),
  tf.keras.layers.Dense(1)
])
critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))

def discount_normalize_rewards(running_add,r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r

env = gym.make('LunarLander-v2')
env.seed(1)
# env._max_episode_steps = 1000
episodes = 1000
score=0
episode_n=[]
mean_score=[]


max_score=200
Actor_update_steps = 10
Critic_update_steps = 10


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

    critic_model.fit(previous_states, real_previous_values, epochs=Critic_update_steps,verbose=0,batch_size=len(buff))
    advantages=np.vstack(advantages)
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages=tf.squeeze(advantages)
    previous_states=np.vstack(previous_states)

    for _ in range(Actor_update_steps):
      with tf.GradientTape() as tape:
        losses=losss(previous_states,actions,advantages)

      grads = tape.gradient(losses, actor_model.trainable_variables)
      # grads, _ = tf.clip_by_global_norm(grads, 0.5)
      optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
    
    
	
for e in range(episodes):
  state = env.reset()
  episode_score = 0
  episode_memory=[]
  done = False
  running_add=0
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
    state=next_state
  episode_memory=np.array(episode_memory)
  # episode_memory[:,2] = discount_normalize_rewards(running_add,episode_memory[:,2])
  train(episode_memory)
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
fig.savefig("ppo_gae.png")
plt.show()