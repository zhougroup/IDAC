python run_idac.py \
--device "cuda:1" \
--pi_type implicit \
--env_name "Walker2d-v2" &
python run_idac.py \
--device "cuda:2" \
--pi_type implicit \
--env_name "HalfCheetah-v2" &
python run_idac.py \
--device "cuda:3" \
--pi_type implicit \
--env_name "Ant-v2"