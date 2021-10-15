id="aoa"
if [ ! -f log/log_$id/infos_$id.pkl ]; then
start_from=""
else
start_from="--start_from log/log_$id"
fi
python train.py --id $id \
    --num_heads 8 \
    --multi_head_scale 1 \
    --ctx_drop 1 \
    --dropout_aoa 0.3 \
    --input_json data/cocotalk.json \
    --input_label_h5 data/cocotalk.h5 \
    --feat_dir data/feats \
    --seq_per_img 5 \
    --batch_size 10 \
    --beam_size 1 \
    --num_layers 2 \
    --input_encoding_size 1024 \
    --rnn_size 1024 \
    --language_eval 1 \
    --val_images_use -1 \
    --save_checkpoint_every 3000 \
    --start_from log/log_$id \
    --checkpoint_path log/log_$id"_rl" \
    --learning_rate 2e-5 \
    --max_epochs 15 \
    --self_critical_after 0 \
    --learning_rate_decay_start -1 \
    --scheduled_sampling_start -1 \
    --reduce_on_plateau


