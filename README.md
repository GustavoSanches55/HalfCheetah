# HalfCheetah
A repository dedicated for learning Reinforcement learning using the model Half Cheetah

> A presentation is avaliable [here](./MuJoco%20aprendizado%20por%20reforço.pdf)

## Installation

First you need to install Mujoco:
https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/
https://github.com/openai/mujoco-py#install-mujoco
Aqui estão os comandos :
```bash	
curl -OL https://www.roboti.us/download/mujoco200_linux.zip

mkdir ~/.mujoco
cp mujoco200_linux.zip ~/.mujoco
cd ~/.mujoco
# extract to ~/.mujoco/mujoco200
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200

```

Then you need to download the license key from the site:
https://www.roboti.us/file/mjkey.txt

```bash
curl -OL https://www.roboti.us/file/mjkey.txt
cp mjkey.txt ~/.mujoco
cp mjkey.txt ~/.mujoco/mujoco200/bin
```

Adding the path to the .bashrc file:
```bash
gedit ~/.bashrc

export LD_LIBRARY_PATH=/home/csy/.mujoco/mujoco200/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export MUJOCO_KEY_PATH=/home/csy/.mujoco${MUJOCO_KEY_PATH}
```

You can also thest Mujoco using the following command:
```bash
cd ~/.mujoco/mujoco200/bin
./simulate ../model/humanoid.xml
```
Se não encontrar libxcursor, instale com:
```bash
sudo apt-get update
sudo apt-get install libxcursor1
```

Caso tenha problemas com o teste, no ubuntu pode ser necessário:
```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```


Install the requirements using pip:

```bash
pip install -r requirements.txt
```



## Usage
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
