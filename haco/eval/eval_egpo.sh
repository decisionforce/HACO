path=""
folder="eval_no_lag"

nohup python ~/drivingforce/drivingforce/human_in_the_loop/eval/evaluate_egpo.py --folder ${folder} --start_ckpt 150 --num_ckpt 150 --path ${path} > ${folder}.log 2>&1 &

#for ((i = 30; i < 299; i += 30)); do
#  nohup python evaluate_egpo.py --folder ${folder} --start_ckpt ${i} --num_ckpt 30 --path ${path} > ${folder}_start_ckpt_${i}.log 2>&1 &
#done
