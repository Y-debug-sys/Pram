GPUs=4
PORT=$((29500 + RANDOM % 1000))

batch_size=16
eval_batch_size=1
num_paths=4
num_layers=8
history_length=12
learning_rate=0.0001
dropout=0.1
normlized_scale=1000000000
hidden_dim=64

#######################################
# Minimizing Maximum Link Utilization #
#######################################

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\
  --topo_fname ./data/topology/GEANT.json\
  --dm_fname ./data/demand/GEANT.csv\
  --is_training 1\
  --num_itrs 3\
  --train_epochs 5\
  --patience 2\
  --d_model $hidden_dim\
  --mllm_name qwen7b\
  --d_mllm '3584'\
  --batch_size $batch_size\
  --eval_batch_size $eval_batch_size\
  --mllm_layers $num_layers\
  --scale $normlized_scale\
  --num_paths $num_paths\
  --history_len $history_length\
  --learning_rate $learning_rate\
  --synthesis 0\
  --objective MLU

########################
# Maximizing Toal Flow #
########################

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\
  --topo_fname ./data/topology/GEANT.json\
  --dm_fname ./data/demand/GEANT.csv\
  --is_training 1\
  --num_itrs 3\
  --train_epochs 5\
  --patience 2\
  --d_model $hidden_dim\
  --mllm_name qwen7b\
  --d_mllm '3584'\
  --batch_size $batch_size\
  --eval_batch_size $eval_batch_size\
  --mllm_layers $num_layers\
  --scale $normlized_scale\
  --num_paths $num_paths\
  --history_len $history_length\
  --learning_rate $learning_rate\
  --synthesis 0\
  --objective MTF

##############################
# Maximizing Concurrent Flow #
##############################

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $GPUs --main_process_port $PORT main.py \
  --topology GEANT\
  --topo_fname ./data/topology/GEANT.json\
  --dm_fname ./data/demand/GEANT.csv\
  --is_training 1\
  --num_itrs 3\
  --train_epochs 5\
  --patience 2\
  --d_model $hidden_dim\
  --mllm_name qwen7b\
  --d_mllm '3584'\
  --batch_size $batch_size\
  --eval_batch_size $eval_batch_size\
  --mllm_layers $num_layers\
  --scale $normlized_scale\
  --num_paths $num_paths\
  --history_len $history_length\
  --learning_rate $learning_rate\
  --synthesis 0\
  --objective MCF
