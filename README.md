# VRP-DACT

DACT is a learning based improvement model for solving routing problems (e.g., TSP and CVRP), which explores dual-aspect representation, dual-aspect collaborative attention (DAC-Att) and cyclic positional encoding (CPE). It is trained by n-step Proximal Policy Optimization (PPO) with a curriculum learning (CL) strategy.

In branch [new_version](https://github.com/yining043/VRP-DACT/tree/new_version), we have fixed some inaccuracies in the CVRP feasibility mask calculation. As a result, we could reduce the training time of CVRP while achieving similar or even better performance. The new pre-trained models are provided in the [pretrained](./pretrained) folder, and the new results are updated in the latest [ArXiv preprint](https://arxiv.org/abs/2110.02544).

![](https://raw.githubusercontent.com/yining043/TSP-improve/master/outputs/ep_gif_0.gif)

# Paper
![architecture](./architecture.jpg)

This repo implements our paper: Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, Jing Tang, “[Learning to iteratively solve routing problems with dual-aspect collaborative transformer](https://arxiv.org/abs/2110.02544),” in Advances in Neural Information Processing Systems, vol. 34, 2021. Please cite our paper if the code is useful for your project.
```
@inproceedings{ma2021learning,
 author = {Ma, Yining and Li, Jingwen and Cao, Zhiguang and Song, Wen and Zhang, Le and Chen, Zhenghua and Tang, Jing},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {11096--11107},
 title = {Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer},
 volume = {34},
 year = {2021}
}
```


# Jupyter Notebook
We provide a Jupyter notebook to help you get started and understand our code. Please open the [Jupyter notebook](./Play_with_DACT.ipynb) for more details.

Please note that in our implementation, the VRP solution is stored in a linked list format. Let us consider a TSP-20 solution [ 6 -> 17 -> 3 -> 9 -> 16 -> 4 -> 12 -> 0 -> 1 -> 5 -> 13 -> 19 -> 11 -> 18 -> 8 -> 14 -> 15 -> 7 -> 2 -> 10 -> 6], we would store this solution as rec = torch.tensor([[ 1, 5, 10, 9, 12, 13, 17, 2, 14, 16, 6, 18, 0, 19, 15, 7, 4, 3,8, 11]]),. Here, if rec[i] = j, it means the node i is connected to node j, i.e., edge i-j is in the solution. For example, edge 0-1, edge 1-5, edge 2-10 are in the solution, so we have rec[0]=1, rec[1]=5 and rec[2]=10.


# One more thing
You may be interested in our new approach called [N2S](https://github.com/yining043/PDP-N2S) (IJCAI 2022) which makes DACT more efficient for solving pickup and delivery problems (PDPs). And the proposed Synthetic Attention (Synth-Att) in N2S has the potential to improve the efficiency of DACT further.

Paper: Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Hongliang Guo, Yuejiao Gong and Yeow Meng Chee, “[Efficient Neural Neighborhood Search for Pickup and Delivery Problems](https://arxiv.org/abs/2204.11399),” in the 31st International Joint Conference on Artificial Intelligence (IJCAI 22), 2022.

# Dependencies
* Python>=3.6 (we test on Python 3.8.2)
* PyTorch>=1.1  (we test on PyTorch 1.7.0)
* numpy
* tensorboard_logger
* tqdm

# Usage
## Generating data
Training data is generated on the fly. Please follow repo [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) to generate validating or test data if needed. We also provide some randomly generated data in the  [datasets](./datasets) folder.

## Training
### TSP example


For training TSP instances with 100 nodes and GPU cards {0}:
```python


# 不预先构造解 64维度 ws bs=256 1个epoch:40batch 每个batch train200步
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --graph_size 100 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size100.pt' --run_name 'direct2improve_tsp_100_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 128 --epoch_size 5120  > direct2improve_tsp_100_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch.txt 


# 不预先构造解 64维度 ws bs=128 1个epoch:40batch 每个batch train200步
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 100 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size100.pt' --run_name 'direct2improve_tsp_100_ws_model_normal_PFE_1_EMBEDDER_Ttrain200_40batch' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 6 --batch_size 128 --epoch_size 5120 --validate_interval 5 > direct2improve_tsp_100_ws_model_normal_PFE_1_EMBEDDER_Ttrain200_40batch.txt 




# 预先构造解 64维度 ws bs=128 T_train=300 --construct_begin_epoch 10
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 100 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size100.pt' --run_name 'construct2improve_example_training_TSP100_ws_normal_model_PFE_2_EMBEDDER_batchsize_128_Ttrain300_epoch10' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 128 --epoch_size 2560 --construct_begin_epoch 10 --with_construct_before > construct2improve_tsp_100_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_epoch10.txt 


```




For training TSP instances with 50 nodes and GPU cards {0}:
```python


#预先构造解 64维度 bs=256
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER__batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --with_construct_before  > construct2improve_tsp_50_ws_normal_model_PFE_2_EMBEDDER.txt 

CUDA_VISIBLE_DEVICES=0 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws' --dec_method 'WS' --with_construct_before > construct2improve_tsp_50_ws.txt

# 128维度 ws bs=256
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_larger_model_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --with_construct_before  > construct2improve_tsp_50_ws_model_larger.txt 


# 预先构造解 64维度 ws bs=256 T_train=500
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 500 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain500' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240  --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain500.txt 

# 预先构造解 64维度 ws bs=256 T_train=300 --construct_begin_epoch 10
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain300_epoch10' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 5120 --construct_begin_epoch 10 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_epoch10.txt 

# 预先构造解 64维度 ws bs=128 T_train=300 --construct_begin_epoch 5
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_128_Ttrain300_epoch5' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 128 --epoch_size 2560 --construct_begin_epoch 5 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_epoch5.txt 

# 预先构造解 64维度 ws bs=256 T_train=300 --construct_begin_epoch 10 带初始参数的improve模型再去更新
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain300_constructepoch10_loadepoch40' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 5120 --construct_begin_epoch 10 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-40.pt' --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_constructepoch10_loadepoch40.txt 

# 预先构造解 64维度 ws bs=256 T_train=300 --construct_begin_epoch 200 带初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain300_constructepoch200_loadepoch40' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 5120 --construct_begin_epoch 200 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-40.pt' --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_constructepoch200_loadepoch40.txt 


# 预先构造解 64维度 ws bs=256 T_train=500 --construct_begin_epoch 0 带初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 500 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain500_constructbegin_0_constructend_20_loadepoch0' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 0 --construct_end_epoch 20 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-0.pt' --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain500_40batch_constructbegin_0_constructend_20_loadepoch0.txt 


# 预先构造解 64维度 ws bs=256 T_train=500 --construct_begin_epoch 0 随机参数开始训练 
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 500 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain500_constructbegin_0_constructend_20_noload' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 0 --construct_end_epoch 20  --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain500_40batch_constructbegin_0_constructend_20_noload.txt 

# 预先构造解 64维度 ws bs=256 T_train=800 --construct_begin_epoch 0 随机参数开始训练 
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 800 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain800_constructbegin_0_constructend_20_noload' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 7680 --construct_begin_epoch 0 --construct_end_epoch 20  --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain800_30batch_constructbegin_0_constructend_20_noload.txt 

# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 0 带初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_0_constructend_20_loadepoch199' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 0 --construct_end_epoch 20 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_0_constructend_20_loadepoch199.txt 


# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 1 --construct_end_epoch 13  -epoch_end 130 --epochs_each_construct_epoch_to_learn 10 带最好direct初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=5 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_1_constructend_13_loadepoch199_total_130epoch' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 1 --construct_end_epoch 13 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --epoch_end 130 --epochs_each_construct_epoch_to_learn 10  --lr_decay 0.997 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_1_constructend_13_loadepoch199_total_130epoch.txt 

# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 1 --construct_end_epoch 13  -epoch_end 260 --epochs_each_construct_epoch_to_learn 20 带最好direct初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_1_constructend_13_loadepoch199_total_260epoch' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 1 --construct_end_epoch 13 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --epoch_end 260 --epochs_each_construct_epoch_to_learn 20  --lr_decay 0.998 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_1_constructend_13_loadepoch199_total_260epoch.txt 

# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 1 --construct_end_epoch 13  -epoch_end 520 --epochs_each_construct_epoch_to_learn 40 带最好direct初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=0 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_1_constructend_13_loadepoch199_total_520epoch' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 1 --construct_end_epoch 13 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --epoch_end 520 --epochs_each_construct_epoch_to_learn 40  --lr_decay 0.999 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_1_constructend_13_loadepoch199_total_520epoch.txt 



# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 1 --construct_end_epoch 13  -epoch_end 130 --epochs_each_construct_epoch_to_learn 10 带最好direct初始参数的improve模型再去更新 --lr_decay 0.984 
CUDA_VISIBLE_DEVICES=0 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_1_constructend_13_loadepoch199_total_130epoch_lr0_984' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 1 --construct_end_epoch 13 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --epoch_end 130 --epochs_each_construct_epoch_to_learn 10  --lr_decay 0.984 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_1_constructend_13_loadepoch199_total_130epoch_lr0_984.txt 



# 预先构造解 64维度 ws bs=256 T_train=200 --construct_begin_epoch 1 --construct_end_epoch 13  -epoch_end 260 --epochs_each_construct_epoch_to_learn 20 --lr_decay 0.989  带最好direct初始参数的improve模型再去更新 
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain200_constructbegin_1_constructend_13_loadepoch199_total_260epoch_lr_decay_0_989' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --construct_begin_epoch 1 --construct_end_epoch 13 --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --epoch_end 260 --epochs_each_construct_epoch_to_learn 20  --lr_decay 0.989 --with_construct_before > construct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER_Ttrain200_40batch_constructbegin_1_constructend_13_loadepoch199_total_260epoch_lr_decay_0_989.txt 









# 不预先构造解 128维度 ws bs=256
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'direct2improve_example_training_TSP50_ws_larger_model_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 3 --batch_size 256 --epoch_size 10240  > direct2improve_tsp_50_ws_model_larger.txt 

# 不预先构造解 64维度 ws bs=256
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 50 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --run_name 'direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240  > direct2improve_tsp_50_ws_model_normal_PFE_2_EMBEDDER.txt 



```

For training TSP instances with 20 nodes and GPU cards {2}:
```python

#预先构造解 64维度 bs=256
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws_normal_model_PFE_2_EMBEDDER_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --with_construct_before > construct2improve_tsp_20_ws_normal_model_PFE_2_EMBEDDER.txt

CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws' --dec_method 'WS' --with_construct_before > construct2improve_tsp_20_ws.txt

CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws_larger_model_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --with_construct_before > construct2improve_tsp_20_ws_model_larger.txt

# 预先构造解 64维度 ws bs=256 T_train=500
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 500 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain500' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240 --with_construct_before > construct2improve_tsp_20_ws_normal_model_PFE_2_EMBEDDER_Ttrain500.txt 

# 预先构造解 64维度 ws bs=256 T_train=300 --construct_begin_epoch 10
CUDA_VISIBLE_DEVICES=2 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain300_epoch10' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 5120 --construct_begin_epoch 10 --with_construct_before > construct2improve_tsp_20_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_epoch10.txt 


# 预先构造解 64维度 ws bs=128 T_train=300 --construct_begin_epoch 5
CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 300 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'construct2improve_example_training_TSP20_ws_normal_model_PFE_2_EMBEDDER_batchsize_128_Ttrain300_epoch5' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 128 --epoch_size 2560 --construct_begin_epoch 5 --with_construct_before > construct2improve_tsp_20_ws_model_normal_PFE_2_EMBEDDER_Ttrain300_20batch_epoch5.txt 








# 不预先构造解 128维度 ws bs=256
CUDA_VISIBLE_DEVICES=4 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'direct2improve_example_training_TSP20_ws_larger_model_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 128 --hidden_dim 128 --n_encode_layers 3 --batch_size 256 --epoch_size 10240  > direct2improve_tsp_20_ws_model_larger.txt 

# 不预先构造解 64维度 ws bs=256
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --graph_size 20 --step_method 2_opt --n_step 4 --T_train 200 --Xi_CL 2 --best_cl --max_grad_norm 0.2 --val_m 1 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size20.pt' --run_name 'direct2improve_example_training_TSP20_ws_normal_model_PFE_2_EMBEDDER_batchsize_256' --dec_method 'WS' --DACTencoder_head_num 4 --DACTdecoder_head_num 4 --embedding_dim 64 --hidden_dim 64 --n_encode_layers 3 --batch_size 256 --epoch_size 10240  > direct2improve_tsp_20_ws_normal_model_PFE_2_EMBEDDER.txt 



```

### CVRP example
For training CVRP instances with 20 nodes and GPU card 0:
```python
CUDA_VISIBLE_DEVICES=0 python run.py --problem vrp --graph_size 20 --dummy_rate 0.5 --step_method 2_opt --n_step 5 --T_train 250 --Xi_CL 1 --best_cl --max_grad_norm 0.04 --val_m 1 --val_dataset  './datasets/cvrp_20_10000.pkl' --run_name 'example_training_CVRP20'
```
Note: hyper-parameters ''--n_step 5 --T_train 250'' are good enough for CVRP now. And we consider loading the pre-trained models of TSP50, CVRP20, and CVRP50 to train TSP100, CVRP-50, and CVRP100 for faster convergency respectively. Please pay attention to the argument "--dummy_rate" for CVRP where we should use different values for different CVRP sizes (e.g., we use 0.5 for CVRP20, 0.4 for CVRP50, and 0.2 for CVRP100).

### Warm start
You can initialize a run using a pretrained model by adding the --load_path option:
```python
--load_path '{add model to load here}'
```
### Resume Traning
You can resume a training by adding the --resume option:
```python
--resume '{add last saved checkpoint(model) to resume here}'
```
The Tensorboard logs will be saved to folder "logs" and the trained model (checkpoint) will be saved to folder "outputs". Pretrained models are provided in the [pretrained](./pretrained) folders.

## Inference
Load the model and specify the iteration T for inference (using --val_m for data augments):

```python
--eval_only 
--load_path '{add model to load here}'
--T_max 10000 
--val_size 10000 
--val_dataset '{add dataset here}' 
--val_m 8
--init_val_met greedy
```

### Examples
For inference 200 TSP instances with 50 nodes and no data augment:
```python
CUDA_VISIBLE_DEVICES=3 python run.py --problem tsp --graph_size 50 --step_method 2_opt --eval_only --init_val_met random --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-5.pt' --T_max 1500 --val_size 200 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --val_m 1 --no_saving --no_tb

CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 50 --step_method 2_opt --eval_only --init_val_met random --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/direct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_20240127T183349/epoch-199.pt' --T_max 1500 --val_size 200 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --val_m 1 --no_saving --no_tb


#预先构造解

CUDA_VISIBLE_DEVICES=1 python run.py --problem tsp --graph_size 50 --step_method 2_opt --eval_only --init_val_met random --load_path '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/construct_model/TSP/outputs/tsp_50/construct2improve_example_training_TSP50_ws_normal_model_PFE_2_EMBEDDER_batchsize_256_Ttrain500_constructbegin_0_constructend_20_loadepoch0_20240201T025601/epoch-1.pt' --T_max 1500 --val_size 200 --val_dataset  '/home/pjy/OPEN_SOURCE_CODE/VRP-DACT/data/testdata_tsp_size50.pt' --val_m 1 --no_saving --no_tb --construct_begin_epoch 1 --with_construct_before











```
For inference 512 CVRP instances with 100 nodes and 8 data augments:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --problem vrp --graph_size 100 --dummy_rate 0.2 --step_method 2_opt --eval_only --init_val_met greedy --load_path 'pretrained/cvrp100-epoch-190.pt' --T_max 10000 --val_size 512 --val_dataset  './datasets/cvrp_100_10000.pkl' --val_m 8 --no_saving --no_tb
```

See [options.py](./options.py) for detailed help on the meaning of each argument.
For generalization of DACT on larger sizes and different distributions, we consider reducing the opts.P to a smaller value (see line 81 of options.py). Meanwhile, when the distribution/size is significantly different, we can consider the way 2 of generalizing CPEs (see lines 544-550 of nets/graph_layers.py) for (not guaranteed) better performance (optional). Please pay attention to the argument "--dummy_rate" for CVRP where we should use different values for different CVRP sizes (e.g., we use 0.5 for CVRP20, 0.4 for CVRP50, and 0.2 for CVRP100 by default).

# Acknowledgements
The code and the framework are based on the repos [wouterkool/attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route) and [yining043/TSP-improve](https://github.com/yining043/TSP-improve). And we thank [@small-Qing](https://github.com/small-Qing) and [@yzhang-gh](https://github.com/yzhang-gh) for raising insightful issues to help improve this project.
