python run_idac.py \
--device "cuda:0" \
--alpha 0.4 \
--use_automatic_entropy_tuning 0 \
--env_name "Hopper-v2" &
python run_idac.py \
--device "cuda:0" \
--alpha 0.3 \
--use_automatic_entropy_tuning 0 \
--env_name "Hopper-v2" &
python run_idac.py \
--device "cuda:1" \
--alpha 0.4 \
--use_automatic_entropy_tuning 0 \
--env_name "Walker2d-v2" &
python run_idac.py \
--device "cuda:1" \
--alpha 0.3 \
--use_automatic_entropy_tuning 0 \
--env_name "Walker2d-v2" &
python run_idac.py \
--device "cuda:2" \
--alpha 0.4 \
--use_automatic_entropy_tuning 0 \
--env_name "Ant-v2" &
python run_idac.py \
--device "cuda:3" \
--alpha 0.2 \
--use_automatic_entropy_tuning 1 \
--env_name "Ant-v2"