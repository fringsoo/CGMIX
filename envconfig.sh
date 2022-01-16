#on iiisct
#/bin/bash
#. /home/wth/miniconda3/etc/profile.d/conda.sh

# conda create -n cgmix python=3.6
# conda activate cgmix

#Change the https to http in ~/.condarc
conda env create -f environment_yql.yml
source activate yql
pip install -r requirements_yql.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip install -r requirements_yqltorch.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple 
#wget https://data.pyg.org/whl/torch-1.7.0%2Bcu102/torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl

#later found out that requirements.txt version is not capable to run the experiment without bugs.
#Directly use yql's conda env.

bash install_sc2.sh
cp -f maps/* ./3rdparty/StarCraftII/Maps/Melee 


# CUDA_VISIBLE_DEVICES=$GPU python3 src/main.py --config=cgmix --env-config=ghost_hunt
# CUDA_VISIBLE_DEVICES=$GPU nohup python3 src/main.py --config=dcg --env-config=sc2 > nohup_dcgrank1duel_5000rnn.out &
# CUDA_VISIBLE_DEVICES=$GPU nohup python3 src/main.py --config=cgmix --env-config=sc2 > nohup_cgmix_real_mixemd2_5000rnn.out &
# CUDA_VISIBLE_DEVICES=6 nohup python3 src/main.py --config=dcg --env-config=pursuit > results/1-nohup_pursuit_dcg_5000rnnfeat.out &
# CUDA_VISIBLE_DEVICES=6 nohup python3 src/main.py --config=cgmix --env-config=pursuit > results/2-nohup_pursuit_cgmix_mixemd1_leaky1_5000rnnfeat.out &

nohup python3 src/main.py --config=cgmix --env-config=toygame >/dev/null 2>&1 &
nohup python3 src/main.py --config=cgmix --env-config=toygame >/dev/null 2>log &
python3 src/main.py --config=dcg --env-config=toygame

tensorboard --logdir=results/tb_logs --port=16006 --bind_all &
pip3 install seaborn==0.9.0