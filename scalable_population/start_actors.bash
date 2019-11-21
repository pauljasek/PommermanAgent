#!/bin/bash
for i in $(seq 0 $(($3 - 1))); do
	python experiment.py --job_name=actor --task=$(($3*$2 + $i)) --num_agents=$1 --processes=$3 --logdir=$4 &
done;
wait

#for i in $(seq 0 $(($1 - 1))); do
#	for j in $(seq 0 $(($1 - 1))); do
#		python experiment.py --job_name=actor --task=$(($1*$1*$2 + $i*$1 + $j)) --agent1=$i --agent2=$j --num_agents=$1 --logdir=agents/distributed &
#  done;
#done;
#wait
