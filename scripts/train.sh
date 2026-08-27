python train.py \
--task_name "samha_run" \
--experiment "exp01" \
--gpu "0,1" \
--input_mode 3 \
--dataset 1 \
--n_class 2 \
--batch_size 2 \
--sub_batch_size 6 \
--size_p 672 \
--size_g 672 \
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
--pre_path "test_samha_weights.pth"

# Supported distance-prior ablations

# 1. Exponential learned (main/best model and default)
# --distance_prior "exp" \
# --lambda_dist_trainable True \

# 2. Exponential fixed
# --distance_prior "exp" \
# --lambda_dist_trainable False \

# 3. Gaussian learned
# --distance_prior "gaussian" \
# --lambda_dist_trainable True \

# 4. Gaussian fixed
# --distance_prior "gaussian" \
# --lambda_dist_trainable False \

# 5. No distance bias (lambda is automatically disabled)
# --distance_prior "none" \
# --lambda_dist_trainable False \
