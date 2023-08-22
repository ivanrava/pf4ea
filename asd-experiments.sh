#!/usr/bin/env bash

seed=42
count=0
for width in {48..100..16}
do
	for height in {48..100..16}
	do
		size=$((width*height))
		for num_agents in {2..4}
		do
			for obstacle_ratio in 0 0.1 0.25 0.5 0.75 0.9
			do
				obstacles=$(echo "print(int($size*$obstacle_ratio))" | python3)
				free_cells=$((size-obstacles))
				maximum_path_length=$((free_cells / 2))
				for conglomeration_ratio in 0 0.25 0.5 0.75 1
				do
					for agent_path_length in 5 25 50 100 500 1000 5000
					do
						if [[ $agent_path_length < $free_cells ]]
						then
							for max_length in 5 10 50 100 500 1000 5000
							do
								if [[ $max_length < $free_cells ]]
								then
									for agent_generator in "random" "optimal"
									do
										for h in "dijkstra" "diagonal"
										do
											count=$((count+1))
										  echo -n "$count;$width;$height;$num_agents;$obstacle_ratio;$conglomeration_ratio;$agent_path_length;$max_length;$agent_generator;$h;$seed;"
											/usr/bin/time 2>&1 -f ";%e;%M" python3 main.py $width $height $num_agents $obstacle_ratio $conglomeration_ratio $agent_path_length $max_length $agent_generator $h $seed
											>&2 echo -en "\r$count... "
										done
									done
								fi
							done
						fi
					done
				done
			done
		done
	done
done
# echo $count
