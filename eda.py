import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


if __name__ == '__main__':
    # Read data
    df = pd.read_csv('42.csv', sep=';')
    df.drop('id', axis=1, inplace=True)
    df.drop('seed', axis=1, inplace=True)

    # Data cleaning
    df.loc[df['num_agents'] == 0, 'agent_path_length'] = None
    df.loc[df['num_agents'] == 0, 'agent_generator'] = None

    df['memory'] = df['memory'] / 1000

    df['resolution_time'] *= 1000
    df['grid_gen_time'] *= 1000
    df['agents_gen_time'] *= 1000
    df['process_time'] *= 1000

    df['size'] = df['width'] * df['height']
    df['obstacles'] = (df['size'] * df['obstacle_ratio']).astype('int')
    df['free_cells'] = df['size'] - df['obstacles']
    print(df)

    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))

    df_memory_groups = df.groupby('size').agg(MemoryMin=('memory', 'min'),
                           MemoryMean=('memory', 'mean'),
                           MemoryMax=('memory', 'max'))

    sns.lineplot(data=df_memory_groups)
    plt.show()
    sns.violinplot(x='size', y='memory', data=df)
    plt.show()




# graphic size - resolution time (filtered on status success)
    df_success = df.loc[df['status'] == 'success']

    df_diagonal = df_success.loc[df['h'] == 'diagonal']
    df_dijkstra = df_success.loc[df['h'] == 'dijkstra']

    #grafico size - process time (filtered on success )
    sns.lineplot(data=df_diagonal.groupby('size').agg(ProcessTimeMeanDiagonal=('process_time','mean')),  palette=['red'])
    sns.lineplot(data=df_dijkstra.groupby('size').agg(ProcessTimeMeanDijkstra=('process_time', 'mean')), palette=['blue'])
    plt.title('size - process time by heuristic on success')
    plt.show()



#histogram count status per obstacle ratio
    obstacle_ratio_groups = df.groupby('obstacle_ratio')['status'].value_counts()
    print(obstacle_ratio_groups)

    # Convertire la Serie risultante in un DataFrame
    df_obstacle_ratio_groups = obstacle_ratio_groups.unstack(fill_value=0)

    print(df_obstacle_ratio_groups)

    df_obstacle_ratio_groups.plot(kind='bar')
    plt.xlabel('obstaclo_ratio')
    plt.ylabel('tot')
    plt.title('Conteggio occorrenze status per obstacle ratio')
    plt.legend(title='Status')
    plt.show()

    print(df.info)

#histogram grid gen time by size
    df_grid_gen_time_group = df.groupby('size').agg(GridGenTimeMean=('grid_gen_time','mean'),
                                                    AgentsGenTimeMean=('agents_gen_time','mean'),
                                                    ResolutionTimeMean=('resolution_time','mean'))
    print(df_grid_gen_time_group)
    sns.lineplot(data=df_grid_gen_time_group)
    plt.title('size - grid gen time')
    plt.show()


#random and optimal agents
    df_agent_random = df.loc[df['agent_generator'] == 'random'].loc[df['status'] != 'timeout']
    df_agent_optimal = df.loc[df['agent_generator'] == 'optimal'].loc[df['status'] != 'timeout']

    df_agent_random_group = df_agent_random.groupby('size').agg(AgentsGenRandomTimeMean=('agents_gen_time', 'mean'))
    df_agent_optimal_group = df_agent_optimal.groupby('size').agg(AgentsGenOptimalTimeMean=('agents_gen_time', 'mean'))

    sns.lineplot(data=df_agent_random_group, palette=['red'])
    sns.lineplot(data=df_agent_optimal_group, palette=['blue'])
    plt.title('time optimal and random agents generator')
    plt.show()

#num agents - time
    df_num_agents_group = df.groupby('num_agents').agg(AgentsGenTime=('agents_gen_time','mean'),
                                                       ResolutionTime=('resolution_time','mean'))
    print(df_grid_gen_time_group)
    g =  sns.lineplot(data=df_num_agents_group)
    g.set(xticks = [0, 1, 2, 3])
    plt.title('num agents - agents gen time')
    plt.show()

#filter agents on random and optimal
    df_num_agents_group_random = df_agent_random.groupby('num_agents').agg(AgentsGenTimeRandom=('agents_gen_time','mean'),
                                                                    ResolutionTimeRandom=('resolution_time','mean'))
    df_num_agents_group_optimal = df_agent_optimal.groupby('num_agents').agg(AgentsGenTimeOptimal=('agents_gen_time', 'mean'),
                                                                           ResolutionTimeOptimal=('resolution_time', 'mean'))

    sns.lineplot(data=df_num_agents_group_random, palette=['red', 'blue'])
    sns.lineplot(data=df_num_agents_group_optimal, palette=['green', 'black'])
    plt.title('num agents - agents gen time')
    plt.show()

    print(df.keys())



# heatmap memory occupation - obstacle
    grid = np.zeros((5,5))
    for i,o in enumerate([0, 0.25, 0.5, 0.75, 0.9]):
         for j,c in enumerate([0, 0.25, 0.5, 0.75, 1]):
              dfoc = df.loc[df['obstacle_ratio'] == o].loc[df['conglomeration_ratio'] == c].loc[df['status'] == 'success']
              grid[i,j] =  np.mean(dfoc['memory'])

    grid[0,1] = grid[0,2] = grid[0,3] = grid[0,4] = grid[0,0]

# Create a heatmap
    sns.set()
    a = sns.heatmap(grid, annot=True, xticklabels=['0%', '25%', '50%', '75%','100%'], yticklabels=['0%', '25%', '50%', '75%','90%'],
                    fmt=".1f", cmap=sns.cm.rocket_r)
    a.invert_yaxis()

# Add title
    plt.xlabel('conglomeration_ratio')
    plt.ylabel('obstacle_ratio')
    plt.title("Heatmap memory occupation - obstacle")
    plt.show()



