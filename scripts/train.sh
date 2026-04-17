python train.py \
--experiment "experiment_name" \
--task_name "task_name" \
--gpu "0,1" \
--input_mode 3 \
--dataset 1 \
--n_class 2 \
--batch_size 2 \
--sub_batch_size 6 \
--size_p 768 \
--size_g 768 \
--context_M 2 \
--context_L 3 \
--patch_overlap 0.20 \
--lr 1e-5 \
--num_epochs 50 \
--wsi_level 3 \
--use_window True \
--distance_prior "exp" \
--distance_sigma 1.0 \
--lambda_dist_init 0.1 \
--lambda_dist_trainable True \
--pre_path "weights.pth" \
--train \
--val

# log-learned.sh (Default/New SAMHA)
# --distance_prior "log" \
# --lambda_dist_trainable True \

# log-fixed.sh
# --distance_prior "log" \
# --lambda_dist_trainable False \

# exp-learned.sh (Original SAMHA)
# --distance_prior "exp" \
# --lambda_dist_trainable True \

# exp-fixed.sh
# --distance_prior "exp" \
# --lambda_dist_trainable False \

# inv-learned.sh
# --distance_prior "inv" \
# --lambda_dist_trainable True \

# inv-fixed.sh
# --distance_prior "inv" \
# --lambda_dist_trainable False \

# gaussian-learned.sh
# --distance_prior "gaussian" \
# --lambda_dist_trainable True \

# gaussian-fixed.sh
# --distance_prior "gaussian" \
# --lambda_dist_trainable False \

# raw-learned.sh
# --distance_prior "raw" \
# --lambda_dist_trainable True \

# raw-fixed.sh
# --distance_prior "raw" \
# --lambda_dist_trainable False \

# None.sh
# --distance_prior "none" \
# --lambda_dist_trainable False \
