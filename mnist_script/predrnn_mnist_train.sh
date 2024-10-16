export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /home/huangzhe/PrenRNN/data/moving-mnist-example/moving-mnist-train.npz \
    --valid_data_paths /home/huangzhe/PrenRNN/data/moving-mnist-example/moving-mnist-valid.npz \
    --save_dir checkpoints/mnist_predrnn \
    --gen_frm_dir results/mnist_predrnn \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 5000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.0002 \
    --lr 0.0003 \
    --batch_size 8 \
    --max_iterations 8000 \
    --display_interval 100 \
    --test_interval 500 \
    --snapshot_interval 500