# Non-Linear Coordination Graphs
The code is based on the the implementations of the ICML 2020 paper "[Deep Coordination Graphs](https://arxiv.org/abs/1910.00091)" by [Wendelin B&ouml;hmer](https://github.com/wendelinboehmer), [Vitaly Kurin](https://yobibyte.github.io) and [Shimon Whiteson](https://github.com/oxwhirl), which is based on the python/pytorch framework [PyMARL](https://github.com/oxwhirl/pymarl) from the [Starcraft Multi-Agent Challenge](https://arxiv.org/abs/1902.04043).

## Installation instructions 
1. Create a conda environment for the experiment and activate the environment
```shell
conda create -n cgmix python=3.7.10
conda activate cgmix
```
2. Install the dependencies
```shell
##basic requirements
pip install -r requirements_cgmix.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
##torch and torch_scatter versions that are tested in my environment
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/cu110/torch_stable.html
pip install https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl
##smac from github
pip install git+https://github.com/oxwhirl/smac.git
```
<!-- 3. Set up StarCraft II and SMAC
```
bash install_sc2.sh
cp -f maps/* ./3rdparty/StarCraftII/Maps/Melee 
``` -->

## Replicate the experiments  
As in the [PyMARL](https://github.com/oxwhirl/pymarl) framework, all experiments are run like this:  
```shell  
python3 src/main.py --config=$ALG --env-config=$ENV
```  
<!-- 
##gdb for c++ code
CUDA_VISIBLE_DEVICES=5 gdb --tui --args python3 src/main.py --config=cgmix --env-config=matrixgame
winheight src -10

###run
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config=cgmix --env-config=pursuit
CUDA_VISIBLE_DEVICES=0 nohup python3 src/main.py --config=cgmix --env-config=pursuit >/dev/null 2>log &
tensorboard --logdir=result/result.pursuit --port=1600 --bind_all &

###check process
ps -eo pid,lstart,etime,cmd | grep pursuit
sudo /sbin/iptables -I INPUT -p tcp --dport 1901 -j ACCEPT

###remove interval saves
#rm -rf [0-4,6-9][0-9][0-9][0-9][0-9][0-9]
#rm -rf 1[1-4,6-9][0-9][0-9][0-9][0-9][0-9]
#rm -rf *[0,5]5[0-9][0-9][0-9][0-9]
#rm -rf 5[0-9][0-9][0-9][0-9] -->


### Parameters for different algorithms (default settings in src/config/default.yaml)
| Algorithms | `$ALG`        | Detailed parameters can be setted in |
| ---------- | ------------- | -------------------------------------|
| CGMIX      | `cgmix`       | src/config/algs/cgmix.yaml (e.g. mixing_embed_dim, onoff_configamount, leaky_alpha, etc) |
| DCG 	     | `dcg`         | src/config/algs/dcg.yaml (e.g. duelling, etc) |
| QMIX       | `qmix`        | src/config/algs/qmix.yaml (e.g. mixing_embed_dim, etc) |

### Parameters for different environments (default settings in src/config/default.yaml)
| Environments | `$ENV`        | Detailed parameters can be setted in |
| ---------- | ------------- | -------------------------------------|
| Aloha     | `aloha`       | src/config/env/aloha.yaml (e.g. local_results_dir, save_model_interval, t_max, lr, mixingnetworkWb, etc) |
| Sensor    | `sensor`       | src/config/env/sensor.yaml (e.g. local_results_dir, save_model_interval, t_max, lr, mixingnetworkWb, etc) |
| Gather     | `gather`       | src/config/env/gather.yaml (e.g. local_results_dir, save_model_interval, t_max, lr, mixingnetworkWb, etc) |
| Hallway    | `hallway`       | src/config/env/hallway.yaml (e.g. local_results_dir, save_model_interval, t_max, lr, mixingnetworkWb, etc) |
| Pursuit     | `pursuit`       | src/config/env/pursuit.yaml (e.g. local_results_dir, save_model_interval, t_max, lr, mixingnetworkWb, etc) |

Results are stored in the "result/env/algo/model,sacred,tblogs/seeds" format and can be `plotted using the sacred json data` or `tb_logs files via tensorboard`.
```
result
├── result.aloha
│   ├── cgmix_mixemb10_alpha0.5_onff16
│   │   ├── models
│   │   │   ├── cgmix__2022-11-03_17-28-27
│   │   │   ├── cgmix__2022-11-03_17-28-30
│   │   │   ├── cgmix__2022-11-03_17-28-32
│   │   │   └── cgmix__2022-11-03_17-28-35
│   │   ├── sacred
│   │   │   ├── 1
│   │   │   ├── 2
│   │   │   ├── 3
│   │   │   ├── 4
│   │   └── tb_logs
│   │       ├── cgmix__2022-11-03_17-28-27
│   │       ├── cgmix__2022-11-03_17-28-30
│   │       ├── cgmix__2022-11-03_17-28-32
│   │       └── cgmix__2022-11-03_17-28-35
```  
After getting the results, you could plot the performances by
```bash
cd plot; python3 plot.py
```
To plot the optimality and efficiency of the iterative optimization on `Aloha`, specify `evaluate:True`, `checkpoint_path`, `iterative_onoff_configamount` in cgmix.yaml and run
```bash
python3 src/main.py --config=cgmix --env-config=aloha
```
## License  
  
Code licensed under the Apache License v2.0
