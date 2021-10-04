python run_idac.py \
--ExpID test \
--device "cuda:0" \
--env_name "Hopper-v2" &
python run_idac.py \
--ExpID test \
--device "cuda:1" \
--env_name "HalfCheetah-v2" &
python run_idac.py \
--ExpID test \
--device "cuda:2" \
--env_name "Ant-v2" &
python run_idac.py \
--ExpID test \
--device "cuda:3" \
--env_name "Walker2d-v2"
