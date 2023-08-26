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


    def aspect_ratio(row):
        if row['width'] >= row['height']:
            return np.round(row['height'] / row['width'], decimals=2)
        else:
            return np.round(row['width'] / row['height'], decimals=2)


    df['aspect_ratio'] = df.apply(aspect_ratio, axis=1)

    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))

    df_no_timeout = df.loc[df['status'] != 'timeout']

    sns.set_theme()

    plt.show()
    # Lineplot - memory against size
    plt.figure(figsize=(8, 6))
    df_memory_groups = df.groupby('size').agg(Best=('memory', 'min'),
                                              Mean=('memory', 'mean'),
                                              Worst=('memory', 'max'))

    sns.lineplot(data=df_memory_groups, markers=True, dashes=False, markersize=8)
    plt.title('Memory consumption w.r.t. grid size', fontsize=18)
    plt.xlabel('Size')
    plt.ylabel('Memory [MB]')
    plt.show()

    # Violinplot - memory against size
    plt.figure(figsize=(16, 8))
    ax = sns.violinplot(x='size', y='memory', data=df, linewidth=0)
    plt.ylim((97, 110))
    sns.despine(offset=10, trim=True)
    plt.title('Memory consumption distribution w.r.t grid size', fontsize=22)
    plt.xlabel('Size')
    plt.ylabel('Memory [MB]')
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.show()

    # graphic size - resolution time (filtered on status success)
    df_success = df.loc[df['status'] == 'success']

    df_diagonal = df_success.loc[df['h'] == 'diagonal']
    df_dijkstra = df_success.loc[df['h'] == 'dijkstra']

    # grafico size - process time (filtered on success )
    sns.lineplot(data=df, x='size', y='process_time', hue='h')
    plt.legend(labels=['Dijkstra relaxed paths', '95% confidence', 'Diagonal distance', '95% confidence'])
    plt.xlabel('Size')
    plt.ylabel('Process time [s]')
    plt.title('Process time w.r.t. heuristics')
    plt.show()

    # dijkstra-diagonal verify with violin split plot
    df_heuristic = df.rename(columns={'h': 'Euristica'}).replace('dijkstra', 'Cammini rilassati').replace('diagonal', 'D. Diagonale')
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.violinplot(data=df_heuristic[df_heuristic['status'] == 'success'], y='solution_cost', split=True, hue='Euristica', x='status', inner=None, linewidth=1, cut=0)
    plt.xticks([])
    plt.title("Costi delle soluzioni")
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    sum_difference = np.sum(np.abs(
        df_heuristic[df_heuristic['Euristica'] == 'Cammini rilassati']['Euristica'] -
        df_heuristic[df_heuristic['Euristica'] == 'D. Diagonale']['Euristica']
    ))
    print(f'Difference between the two heuristic: {sum_difference} (should be 0)')

    plt.figure(figsize=(6, 7))
    ax = sns.countplot(data=df_heuristic, x="Euristica", hue="status")
    for container in ax.containers:
        ax.bar_label(container=container)
    plt.ylabel('')
    plt.xlabel('')
    plt.title('Stati per euristica')
    plt.legend(title='Stati')
    plt.show()

    # histogram count status per obstacle ratio
    obstacle_ratio_groups = df.groupby('obstacle_ratio')['status'].value_counts()
    print(obstacle_ratio_groups)

    # Convertire la Serie risultante in un DataFrame
    df_obstacle_ratio_groups = obstacle_ratio_groups.unstack(fill_value=0)

    print(df_obstacle_ratio_groups)

    g = df_obstacle_ratio_groups.plot(kind='bar')
    g.set_xticklabels(labels=['0%', '25%', '50%', '75%', '90%'], rotation=0)
    plt.xticks(rotation=0)
    plt.xlabel('Obstacles')
    plt.ylabel('')
    plt.title('Status count w.r.t. obstacle ratio')
    plt.legend(title='Status')
    plt.show()

    # histogram grid gen time by size
    df_grid_gen_time_group = df_no_timeout.groupby('size').agg(GridGenTimeMean=('grid_gen_time', 'mean'),
                                                               AgentsGenTimeMean=('agents_gen_time', 'mean'),
                                                               ResolutionTimeMean=('resolution_time', 'mean'))
    sns.lineplot(data=df_grid_gen_time_group, errorbar=None, markers=True, dashes=False, markersize=6)
    plt.legend(labels=['Grid generation', 'Agents generation', 'Resolution time'])
    plt.title('Generation and solver elapsed times')
    plt.show()

    # random and optimal agents
    df_agent_random = df.loc[df['agent_generator'] == 'random'].loc[df['status'] != 'timeout']
    df_agent_optimal = df.loc[df['agent_generator'] == 'optimal'].loc[df['status'] != 'timeout']

    df_agent_random_group = df_agent_random.groupby('size').agg(AgentsGenRandomTimeMean=('agents_gen_time', 'mean'))
    df_agent_optimal_group = df_agent_optimal.groupby('size').agg(AgentsGenOptimalTimeMean=('agents_gen_time', 'mean'))

    sns.lineplot(data=df_no_timeout, x='size', y='agents_gen_time', hue='agent_generator',
                 errorbar=None, markers=True, dashes=False, markersize=8)
    plt.legend(labels=['Pseudo-random', 'Through ReachGoal'])
    plt.title('Agent generators time comparison')
    plt.xlabel('Size')
    plt.ylabel('Agents generation time')
    plt.show()

    # num agents - time
    df_num_agents_group = df.groupby('num_agents').agg(AgentsGenTime=('agents_gen_time', 'mean'),
                                                       ResolutionTime=('resolution_time', 'mean'))
    g = sns.lineplot(data=df_num_agents_group, errorbar=None, markers=True, dashes=False, markersize=7)
    plt.legend(labels=['Agent generation time', 'Resolution time'])
    g.set(xticks=[0, 1, 2, 3])
    plt.title('Times w.r.t. # of agents')
    plt.xlabel('# of agents')
    plt.ylabel('[s]')
    plt.show()

    # filter agents on random and optimal
    df_num_agents_group_random = df_agent_random.groupby('num_agents').agg(
        AgentsGenTimeRandom=('agents_gen_time', 'mean'),
        ResolutionTimeRandom=('resolution_time', 'mean'))
    df_num_agents_group_optimal = df_agent_optimal.groupby('num_agents').agg(
        AgentsGenTimeOptimal=('agents_gen_time', 'mean'),
        ResolutionTimeOptimal=('resolution_time', 'mean'))

    sns.lineplot(data=df_num_agents_group_random, palette=['red', 'blue'])
    sns.lineplot(data=df_num_agents_group_optimal, palette=['green', 'black'])
    plt.title('# of agents & times')
    plt.xlabel('Times against agent generation criterias')
    plt.ylabel('[s]')
    plt.show()

    # heatmap memory occupation - obstacle
    grid = np.zeros((5, 5))
    for i, o in enumerate([0, 0.25, 0.5, 0.75, 0.9]):
        for j, c in enumerate([0, 0.25, 0.5, 0.75, 1]):
            dfoc = df.loc[df['obstacle_ratio'] == o].loc[df['conglomeration_ratio'] == c].loc[df['status'] == 'success']
            grid[i, j] = np.mean(dfoc['memory'])

    grid[0, 1] = grid[0, 2] = grid[0, 3] = grid[0, 4] = grid[0, 0]

    # Create a heatmap
    sns.set()
    a = sns.heatmap(grid, annot=True, xticklabels=['0%', '25%', '50%', '75%', '100%'],
                    yticklabels=['0%', '25%', '50%', '75%', '90%'],
                    fmt=".1f", cmap=sns.cm.rocket_r)
    a.invert_yaxis()

    # Add title
    plt.title("Memory occupation heatmap")
    plt.xlabel('Conglomeration ratio')
    plt.ylabel('Obstacle ratio')
    plt.show()

    # open closed states
    sns.lineplot(data=df_diagonal.groupby('size').agg(InsertedStates=('inserted_states', 'mean')), palette=['red'])
    sns.lineplot(data=df_dijkstra.groupby('size').agg(ClosedStates=('closed_states', 'mean')), palette=['blue'])
    plt.yscale('log')
    plt.title('States inserted & closed by ReachGoal')
    plt.xlabel('Size')
    plt.ylabel('# of states')
    plt.show()

    # aspect ratio plot
    df_without_timeout = df.loc[df['status'] != 'timeout']
    process_time_aspect_ratio_groups = df_without_timeout.groupby('aspect_ratio')['status'].value_counts()
    df_process_time_aspect_ratio_groups = process_time_aspect_ratio_groups.unstack(fill_value=0)
    success = df_process_time_aspect_ratio_groups['success'] / (
            df_process_time_aspect_ratio_groups['success'] + df_process_time_aspect_ratio_groups['failure'])
    failure = df_process_time_aspect_ratio_groups['failure'] / (
            df_process_time_aspect_ratio_groups['success'] + df_process_time_aspect_ratio_groups['failure'])
    df_process_time_aspect_ratio_groups['success'] = success
    df_process_time_aspect_ratio_groups['failure'] = failure

    df_process_time_aspect_ratio_groups.plot(kind='bar')
    plt.xlabel('aspect_ratio')
    plt.ylabel('tot')
    plt.title('Conteggio occorrenze status per aspect ratio')
    plt.legend(title='Status')
    plt.show()

    # plot line
    sns.lineplot(data=df_success.groupby('aspect_ratio').agg(InsertedStates=('inserted_states', 'mean')), palette=['gray'])
    sns.lineplot(data=df_success.groupby('aspect_ratio').agg(ClosedStates=('closed_states', 'mean')), palette=['black'])
    plt.title('Stati inseriti / chiusi per rapporto d\'aspetto')
    plt.show()

    # dijkstra diagonale states
    sns.lineplot(data=df_diagonal.groupby('size').agg(InsertedStatesDiagonal=('inserted_states', 'mean')),
                 palette=['red'])
    sns.lineplot(data=df_diagonal.groupby('size').agg(ClosedStatesDiagonal=('closed_states', 'mean')), palette=['blue'])
    sns.lineplot(data=df_dijkstra.groupby('size').agg(InsertedStatesDijkstra=('inserted_states', 'mean')),
                 palette=['green'])
    sns.lineplot(data=df_dijkstra.groupby('size').agg(ClosedStatesDijkstra=('closed_states', 'mean')),
                 palette=['black'])
    plt.yscale('log')
    plt.title('size - states inserted/closed Dijkstra Diagonal')
    plt.show()

    # mosse wait
    sns.lineplot(data=df.groupby('size').agg(Waits=('waits', 'mean')))
    plt.title('# of wait moves')
    plt.xlabel('Size')
    plt.show()

    # heatmap width-height
    grid = np.zeros((10, 10))
    for i, h in enumerate(range(10, 101, 10)):
        for j, w in enumerate(range(10, 101, 10)):
            dfoc = df.loc[df['width'] == w].loc[df['height'] == h]
            grid[i, j] = np.mean(dfoc['memory'])

    # Create a heatmap
    sns.set()
    a = sns.heatmap(grid, xticklabels=list(range(10, 101, 10)),
                    yticklabels=list(range(10, 101, 10)),
                    fmt=".1f", cmap=sns.cm.rocket_r, cbar_kws={'label': 'MB'})
    a.invert_yaxis()
    plt.title("Memoria occupata rispetto alle dimensioni della griglia")
    plt.xlabel('Larghezza')
    plt.ylabel('Altezza')
    plt.show()

    # heatmap width-height
    def build_grid(param):
        grid = np.zeros((10, 10))
        for i, h in enumerate(range(10, 101, 10)):
            for j, w in enumerate(range(10, 101, 10)):
                dfoc = df.loc[df['width'] == w].loc[df['height'] == h]
                if param != 'process_time':
                    dfoc = dfoc.loc[dfoc['status'] == 'success']
                grid[i, j] = np.mean(dfoc[param])
        return grid

    # Create a heatmap
    sns.set()
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.suptitle("Tempi di esecuzione [ms] in funzione delle dimensioni della griglia")
    a = sns.heatmap(build_grid('process_time'), xticklabels=list(range(10, 101, 10)),
                    yticklabels=list(range(10, 101, 10)),
                    fmt=".1f", cmap=sns.cm._crest_lut)
    a.invert_yaxis()
    plt.title("Tempo di processo")
    # plt.xlabel('Larghezza')
    plt.ylabel('Altezza')
    plt.subplot(2, 2, 2)
    a = sns.heatmap(build_grid('grid_gen_time'), xticklabels=list(range(10, 101, 10)),
                    yticklabels=list(range(10, 101, 10)),
                    fmt=".1f", cmap=sns.cm._crest_lut)
    a.invert_yaxis()
    plt.title("Tempo di generazione della griglia")
    # plt.xlabel('Larghezza')
    # plt.ylabel('Altezza')
    plt.subplot(2, 2, 3)
    a = sns.heatmap(build_grid('agents_gen_time'), xticklabels=list(range(10, 101, 10)),
                    yticklabels=list(range(10, 101, 10)),
                    fmt=".1f", cmap=sns.cm._crest_lut)
    a.invert_yaxis()
    plt.title("Tempo di generazione degli agenti")
    plt.xlabel('Larghezza')
    plt.ylabel('Altezza')
    plt.subplot(2, 2, 4)
    a = sns.heatmap(build_grid('resolution_time'), xticklabels=list(range(10, 101, 10)),
                    yticklabels=list(range(10, 101, 10)),
                    fmt=".1f", cmap=sns.cm._crest_lut)
    a.invert_yaxis()
    plt.title("Tempo di risoluzione")
    plt.xlabel('Larghezza')
    # plt.ylabel('Altezza')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

