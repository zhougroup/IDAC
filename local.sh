python run_idac.py \
--device "cuda:0" \
--alpha 0.3 \
--use_automatic_entropy_tuning 0 \
--pi_type implicit
--env_name "Hopper-v2" &
python run_idac.py \
--device "cuda:1" \
--alpha 0.2 \
--use_automatic_entropy_tuning 1 \
--pi_type implicit
--env_name "Walker2d-v2" &
python run_idac.py \
--device "cuda:1" \
--alpha 0.2 \
--use_automatic_entropy_tuning 1 \
--pi_type implicit
--env_name "HalfCheetah-v2" &
python run_idac.py \
--device "cuda:2" \
--alpha 0.2 \
--use_automatic_entropy_tuning 1 \
--pi_type implicit
--env_name "Ant-v2" &
python run_idac.py \
--device "cuda:3" \
--alpha 0.2 \
--use_automatic_entropy_tuning 1 \
--pi_type implicit
--env_name "Humanoid-v2"