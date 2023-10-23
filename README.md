# HalfCheetah
A repository dedicated for learning Reinforcement learning using the model Half Cheetah

To train the model, run train.py.
There are some possible arguments:
 --model: Name of the file that will be saved with the model. (by default, it is DDPG_cheetah)
 --render (flag): A flag that will render the training. (by default, it is False, and in that case a video will be rendered at the end)
 --episodes: Number of episodes that the training will have. (by default, it is 200)
 --steps: Number of steps that each episode will have. (by default, it is 1000)
 --batch: Number of samples that will be used in each training step. (by default, it is 128)

Example of use (a model named test_model will be saved, the training will be rendered, 100 episodes will be used, each episode will have 1000 steps and each training step will use 128 samples):

```bash
python train.py --model test_model --render --episodes 100 --steps 1000 --batch 128
```

for testing the model, use the notebook test.ipynb. Just run the cells to see the graphs of the model's performance and to see the model in action.