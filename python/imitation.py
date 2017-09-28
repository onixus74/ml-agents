# # Unity ML Agents
# ## Imitation Learning

from docopt import docopt

import os
from ppo.models import *
from ppo.trainer import ImitationTrainer
from unityagents import UnityEnvironment

_USAGE = '''
Usage:
  imitation (<env>) [options] 

Options:
  --help                     Show this message.
  --max-step=<n>             Maximum number of steps to run environment [default: 5e6].
  --run-path=<path>          The sub-directory name for model and summary statistics [default: ppo].
  --load                     Whether to load the model or randomly initialize [default: False].
  --train                    Whether to train model, or only run inference [default: True].
  --summary-freq=<n>         Frequency at which to save training statistics [default: 64].
  --save-freq=<n>            Frequency at which to save model [default: 100].
  --beta=<n>                 Strength of entropy regularization [default: 1e-3].
  --num-epoch=<n>            Number of gradient descent steps per batch of experiences [default: 5].
  --epsilon=<n>              Acceptable threshold around ratio of old and new policy probabilities [default: 0.2].
  --buffer-size=<n>          How large the experience buffer should be before gradient descent [default: 64].
  --learning-rate=<rate>     Model learning rate [default: 3e-4].
  --hidden-units=<n>         Number of units in hidden layer [default: 64].
  --batch-size=<n>           How many experiences per gradient descent update step [default: 64].
  --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
'''

options = docopt(_USAGE)
print(options)

# General parameters
max_steps = float(options['--max-step'])
model_path = './models/{}'.format(str(options['--run-path']))
summary_path = './summaries/{}'.format(str(options['--run-path']))
load_model = options['--load']
train_model = options['--train']
summary_freq = int(options['--summary-freq'])
save_freq = int(options['--save-freq'])
env_name = options['<env>']
keep_checkpoints = int(options['--keep-checkpoints'])

# Algorithm-specific parameters for tuning
beta = float(options['--beta'])
num_epoch = int(options['--num-epoch'])
epsilon = float(options['--epsilon'])
buffer_size = int(options['--buffer-size'])
learning_rate = float(options['--learning-rate'])
hidden_units = int(options['--hidden-units'])
batch_size = int(options['--batch-size'])

env = UnityEnvironment(file_name=env_name)
print(str(env))
player_brain = env.brain_names[1]
agent_brain = env.brain_names[0]

tf.reset_default_graph()

# Create the Tensorflow model graph
imi_model = create_agent_model(env, lr=learning_rate,
                               h_size=hidden_units, epsilon=epsilon,
                               beta=beta, mode="imitation")

is_continuous = (env.brains[player_brain].action_space_type == "continuous")
use_observations = (env.brains[player_brain].number_observations > 0)
use_states = (env.brains[player_brain].state_space_size > 0)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=keep_checkpoints)

with tf.Session() as sess:
    # Instantiate model parameters
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)
    steps = sess.run(imi_model.global_step)
    summary_writer = tf.summary.FileWriter(summary_path)
    infos = env.reset(train_mode=False)
    player_info = infos[player_brain]
    agent_info = infos[agent_brain]
    trainer = ImitationTrainer(imi_model, sess, agent_info, is_continuous, use_observations, use_states)
    while steps <= max_steps or not train_model:
        if env.global_done:
            infos = env.reset(train_mode=False)
            player_info = infos[player_brain]
            agent_info = infos[agent_brain]
        # Decide and take an action
        new_info = trainer.take_action(player_info, agent_info, env, agent_brain, player_brain)
        infos = new_info
        player_info = infos[player_brain]
        agent_info = infos[agent_brain]
        if len(trainer.training_buffer['actions']) > buffer_size and train_model:
            # Perform gradient descent with experience buffer
            trainer.update_model(batch_size)
        if steps % summary_freq == 0 and steps != 0 and train_model:
            # Write training statistics to tensorboard.
            trainer.write_summary(summary_writer, steps)
        if steps % save_freq == 0 and steps != 0 and train_model:
            # Save Tensorflow model
            save_model(sess, model_path=model_path, steps=steps, saver=saver)
        steps += 1
        sess.run(imi_model.increment_step)
    # Final save Tensorflow model
    if steps != 0 and train_model:
        save_model(sess, model_path=model_path, steps=steps, saver=saver)
env.close()
export_graph(model_path, env_name)
