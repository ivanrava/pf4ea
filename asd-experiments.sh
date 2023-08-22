#!/usr/bin/env bash

seed=42
count=0
echo 'id;width;height;num_agents;obstacle_ratio;conglomeration_ratio;agent_path_length;max_length;agent_generator;h;seed;status;solution_length;solution_cost;closed_states;inserted_states;waits;grid_gen_time;agents_gen_time;resolution_time;process_time;memory'
for width in {12..100..8}
do
	for height in {12..100..8}
	do
		size=$((width*height))
		for num_agents in 0 1 2 4
		do
			for obstacle_ratio in 0 0.25 0.5 0.75 0.9
			do
				obstacles=$(echo "print(int($size*$obstacle_ratio))" | python3)
				free_cells=$((size-obstacles))
				for conglomeration_ratio in 0 0.25 0.5 0.75 1
				do
					for agent_path_length in 5 25 50 100 500 1000 5000
					do
						if (( $agent_path_length < $free_cells ))
						then
							for max_length in 5 10 50 100 500 1000 5000
							do
								if (( $max_length < $free_cells ))
								then
								  status=0;
									for agent_generator in "random" "optimal"
									do
										for h in "dijkstra" "diagonal"
										do
											count=$((count+1))
										  echo -n "$count;$width;$height;$num_agents;$obstacle_ratio;$conglomeration_ratio;$agent_path_length;$max_length;$agent_generator;$h;$seed;"
											/usr/bin/time -q 2>&1 -f ";%e;%M" python3 main.py $width $height $num_agents $obstacle_ratio $conglomeration_ratio $agent_path_length $max_length $agent_generator $h $seed
											status=$?
											>&2 echo -en "\r$count: $status... "
										done
										if [[ $num_agents == 0 ]]
										then
										  break
										fi
									done
									if [[ $status != 1 ]]
									then
									  # Do not increment the max length if last run was not a failure
									  break
									fi
								fi
							done
						fi
						if [[ $num_agents == 0 ]]
						then
						  break
						fi
					done
					if [[ $obstacle_ratio == 0 ]]
					then
					  # Why should we try new conglomerations without obstacles?
					  break
					fi
				done
			done
		done
	done
done
# echo $count
