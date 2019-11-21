#!/bin/bash
python experiment.py --num_agents=$1 --task=0 --job_name=learner --processes=$2 --batch_size=$3 --entropy_cost=$4 --learning_rate=$5 --logdir=$6 &

#for i in $(seq 0 $(($1 - 1))); do
#	python experiment.py --num_agents=$1 --task=$i --job_name=learner --processes=$2 --batch_size=$3 --entropy_cost=$4 --learning_rate=$5 --logdir=$6 &
#done;
#wait