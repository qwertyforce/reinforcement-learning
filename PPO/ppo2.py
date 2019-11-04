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
e_clip=0.2
@tf.function
def losss(states,actions,advantages):
   indices=[]
   for x in range(len(states)):
     indices.append([x,actions[x]])

   logits=actor_model(states)

   probs=tf.math.log(tf.gather_nd(logits,tf.convert_to_tensor(indices)))
   
   logits2=old_actor_model(states)
   old_probs=tf.math.log(tf.gather_nd(logits2,tf.convert_to_tensor(indices)))
   ratios = tf.exp(probs-old_probs)
   # print(probs)
   # print(old_probs)
   # exit()
   # ratios = probs / (old_probs+ 1e-8)
   clip_probs = tf.clip_by_value(ratios, 1.-e_clip, 1.+e_clip)
   # exit()
   # print(tf.minimum(tf.multiply(ratio, advantage), tf.multiply(clip_prob, advantage)))
   # print(advantages)
   # print(-tf.minimum(tf.multiply(ratios, advantages), tf.multiply(clip_probs, advantages)))
   # print("WTDF")
   return -tf.reduce_mean(tf.minimum(tf.multiply(ratios, advantages), tf.multiply(clip_probs, advantages)))

# actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001))
# print(actor_model.trainable_variables)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
old_actor_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu', trainable=False),
  tf.keras.layers.Dense(2, activation='softmax', trainable=False)
])

# x=losss(np.array([[1,1,1,1],[2,2,2,2]]),1,3)
# print(x)
# exit()
def upd_old_policy():
  weights_actor_model = actor_model.get_weights()
  old_actor_model.set_weights(weights_actor_model)


critic_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128,input_shape=(1,4),activation='relu'),
  tf.keras.layers.Dense(1)
])
critic_model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.005))

def discount_normalize_rewards(dones,r, gamma = 0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if(dones[t]==1):
          running_add=0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        # print(discounted_r)
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)+1e-8
    return discounted_r
    

env = gym.make('CartPole-v0')
env.seed(1)
# env._max_episode_steps = 1000
episodes = 500
score=0
episode_n=[]
mean_score=[]
discount_factor=0.99
tau=0.95
max_score=200
batch_size=128
Actor_update_steps = 10
Critic_update_steps = 10


def train(buff):
    upd_old_policy()
    previous_states= []
    real_previous_values=[]
    advantages=[]
    actions=[]

    for previous_state, action, reward, current_state, done in buff:
        actions.append(action)
        previous_states.append(previous_state)
        previous_state_predicted_value=critic_model(previous_state)
        if not done:
            current_state_predicted_value=critic_model(current_state)
        else:
            current_state_predicted_value=0
        real_previous_value = reward + discount_factor * current_state_predicted_value
        real_previous_values.append(real_previous_value)
        advantage=real_previous_value - previous_state_predicted_value
        advantages.append(advantage)
    
 

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
      grads, _ = tf.clip_by_global_norm(grads, 0.5)
      optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
    
episode_memory=[]    
dones=[]
for e in range(episodes):
  state = env.reset()
  episode_score = 0
  done = False
  running_add=0
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

 

    episode_memory.append([state, a, reward, next_state, done])
    if (len(episode_memory)==batch_size):
      episode_memory=np.array(episode_memory)
      episode_memory[:,2] = discount_normalize_rewards(dones,episode_memory[:,2])
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
fig.savefig("ppo.png")
plt.show()