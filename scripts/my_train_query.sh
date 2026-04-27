res=256
lr=1e-5
wr=0
bs=1
shot=1
gas=4 # bs = 8 * bs * gas
step=15000
gpu='0'
seed=42

dataname='CHAOST2'  # 'CHAOST2' or 'SABS'
setting=1  # 1 or 2
supp_idx=2  # 训练的可视化中，选择第二个患者作为支持集
output_dir=./logs/${dataname}_setting${setting}_res${res}_lr${lr}_step${step}_bs${bs}_gpu${gpu}_seed${seed}

pretrain_model="./weight/stable-diffusion-1-5-ref8inchannels-tag8inchannels"  # TODO
dinov2_model="../.cache/huggingface/hub/models--facebook--dinov2-base/snapshots/f9e44c814b77203eaa57a6bdbbd535f21ede1415"  # TODO


# 五折
for fold in 0 1 2 3 4
do
PREFIX="${output_dir}/cv${fold}"
echo $PREFIX
CUDA_VISIBLE_DEVICES=${gpu}  accelerate launch --num_processes 1 --main_process_port $RANDOM --mixed_precision "no" --num_machines 1 \
train_tools/my_train_query.py \
 --benchmark='medical' \
 --mixed_precision="no" \
 --train_batch_size=${bs} \
 --checkpointing_steps 15000 \
 --pretrained_model_name_or_path=${pretrain_model} \
 --pretrained_dinov2_path=${dinov2_model} \
 --scheduler_load_path ./scheduler_1.0_1.0 \
 --output_dir=${PREFIX} \
 --train_data_dir "/data/lmh/dataset" \
 --resolution=${res} \
 --learning_rate=${lr} \
 --lr_warmup_steps ${wr} \
 --max_train_steps=${step} \
 --validation_steps 500 \
 --lr_scheduler polynomial \
 --lr_scheduler_power 1.0 \
 --gradient_accumulation_steps=${gas} \
 --enable_xformers_memory_efficient_attention \
 --max_grad_norm=1.0 \
 --adam_weight_decay=1e-2 \
 --tracker_project_name sd21_train_dis \
 --seed=${seed} \
 --image_ref_column img_ref \
 --image_tag_column img_tag \
 --conditioning_image_ref_column ref_conditioning_image \
 --conditioning_image_tag_column tag_conditioning_image \
 --caption_column 'text' \
 --cache_dir './cache' \
 --allow_tf32 \
 --dataloader_num_workers=4 \
 --checkpoints_total_limit 10 \
 --nshot ${shot} \
 --fold=${fold} \
 --supp_idx=${supp_idx} \
 --validation_images=${dataname} \
 --r_threshold 0.25 \
 --setting=${setting}
done