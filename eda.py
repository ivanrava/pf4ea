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
    df['states_difference'] = df['inserted_states'] - df['closed_states']

    df_success = df.loc[df['status'] == 'success']

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

    # Lineplot - memory against size
    plt.figure(figsize=(8, 6))
    df_memory_groups = df.groupby('size').agg(Best=('memory', 'min'),
                                              Mean=('memory', 'mean'),
                                              Worst=('memory', 'max'))

    sns.lineplot(data=df_memory_groups, markers=True, dashes=False, markersize=8)
    plt.title('Occupazione spaziale rispetto alla dimensione della griglia', fontsize=18)
    plt.xlabel('Dimensione della griglia')
    plt.ylabel('Memoria occupata [MB]')
    plt.show()

    # Violinplot - memory against size
    plt.figure(figsize=(16, 8))
    ax = sns.violinplot(x='size', y='solution_length', data=df_success, linewidth=0)
    sns.despine(offset=10, trim=True)
    plt.title('', fontsize=22)
    plt.xlabel('Size')
    plt.ylabel('Memory [MB]')
    plt.grid(zorder=0)
    ax.set_axisbelow(True)
    plt.show()

    # graphic size - resolution time (filtered on status success)
    sns.lineplot(data=df_success, x='size', y='states_difference')
    plt.title('Stati inseriti / chiusi per rapporto d\'aspetto')
    plt.xlabel('Dimensione della griglia')
    plt.ylabel('Differenza tra stati inseriti e chiusi')
    plt.show()

    df_diagonal = df_success.loc[df['h'] == 'diagonal']
    df_dijkstra = df_success.loc[df['h'] == 'dijkstra']

    # grafico size - process time (filtered on success )
    plt.axhline(y=1000, color='#5559', linestyle='--')
    sns.lineplot(data=df, x='size', y='process_time', hue='h')
    plt.legend(labels=['Orizzonte di timeout', 'Cammini rilassati', '95% confidenza', 'D. Diagonale', '95% confidenza'])
    plt.xlabel('Dimensione della griglia')
    plt.ylabel('Tempo di processo [ms]')
    plt.title('Tempo di processo mediato per euristica')
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
        df_heuristic[df_heuristic['Euristica'] == 'Cammini rilassati']['solution_cost'] -
        df_heuristic[df_heuristic['Euristica'] == 'D. Diagonale']['solution_cost']
    ))
    print(f'Difference between the two heuristics (costs): {sum_difference} (should be 0)')


    plt.figure(figsize=(6, 6))
    sns.violinplot(data=df_heuristic[df_heuristic['status'] == 'success'], y='solution_length', split=True, hue='Euristica', x='status', linewidth=1, cut=0, inner='quartile')
    plt.xticks([])
    plt.title("Lunghezza delle soluzioni")
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

    sum_difference = np.sum(np.abs(
        df_heuristic[df_heuristic['Euristica'] == 'Cammini rilassati']['solution_length'] -
        df_heuristic[df_heuristic['Euristica'] == 'D. Diagonale']['solution_length']
    ))
    print(f'Difference between the two heuristics (lengths): {sum_difference} (should be 0)')

    sns.histplot(data=df_success, x='solution_length')
    plt.title("Lunghezza delle soluzioni")
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

    plt.figure(figsize=(6, 6))
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

    plt.legend(labels=['Pseudo-casuale', 'Con ReachGoal'])
    plt.title('Confronto tra generatori di agenti')
    plt.xlabel('Dimensione della griglia')
    plt.ylabel('Tempo impiegato [ms]')
    plt.show()

    # num agents - time
    sns.set_theme()
    df_num_agents_group = df.groupby('num_agents').agg(AgentsGenTime=('agents_gen_time', 'mean'),
                                                       ResolutionTime=('resolution_time', 'mean'))
    sns.lineplot(data=df_success, x='num_agents', y='agents_gen_time', marker='.', markers=True, markersize=12)
    g = sns.lineplot(data=df_success, x='num_agents', y='resolution_time', marker='.', markers=True, markersize=12)
    plt.legend(labels=['Tempo di generazione degli agenti', '95% confidenza', 'Tempo di risoluzione', '95% confidenza'])
    g.set(xticks=[0, 1, 2, 3])
    plt.title('Tempistiche dipendenti dal numero di agenti')
    plt.xlabel('Numero di agenti')
    plt.ylabel('Tempo [ms]')
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
            dfoc = df.loc[df['obstacle_ratio'] == o].loc[df['conglomeration_ratio'] == c]
            grid[i, j] = np.mean(dfoc['memory'])

    grid[0, 1] = grid[0, 2] = grid[0, 3] = grid[0, 4] = grid[0, 0]

    # Create a heatmap
    sns.set()
    a = sns.heatmap(grid, annot=True, xticklabels=['0%', '25%', '50%', '75%', '100%'],
                    yticklabels=['0%', '25%', '50%', '75%', '90%'],
                    fmt=".1f", cmap=sns.cm.rocket_r)
    a.invert_yaxis()

    # Add title
    plt.title("Memoria occupata rispetto agli ostacoli")
    plt.xlabel('Conglomerazione')
    plt.ylabel('Percentuale di ostacoli')
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
    # sns.lineplot(data=df_success.groupby('aspect_ratio').agg(InsertedStates=('inserted_states', 'mean')), palette=['gray'])
    # sns.lineplot(data=df_success.groupby('aspect_ratio').agg(ClosedStates=('closed_states', 'mean')), palette=['black'])

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

    #stati chiusi per euristica
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    sns.lineplot(data=df_diagonal, x='size', y='closed_states')
    plt.xlabel("Dimensione della griglia")
    plt.ylabel("Stati chiusi")
    plt.title('Diagonale - stati chiusi')
    plt.subplot(1,2,2)
    sns.lineplot(data=df_dijkstra, x='size', y='closed_states')
    plt.title('Cammini rilassati - stati chiusi')
    plt.ylim((0,6))
    plt.xlabel("Dimensione della griglia")
    plt.ylabel("")
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    # better waits chart
    sns.lineplot(data=df_success, x='size', y='solution_length')
    sns.lineplot(data=df_success, x='size', y='waits')
    sns.rugplot(data=df, x='waits')
    plt.xlim((0, 1000))
    plt.show()


    sns.lineplot(data=df.groupby('size').agg(Waits=('waits', 'max')))
    sns.rugplot(data=df.groupby('size').agg(Waits=('waits', 'sum')), x='Waits')
    plt.title('# of wait moves')
    plt.xlabel('Size')
    plt.show()


    sns.scatterplot(data=df_success, x='size', y='solution_length', hue='waits')
    plt.show()

    # heatmap width-height
    grid = np.zeros((5, 5))
    for i, o in enumerate([0, 0.25, 0.5, 0.75, 0.9]):
        for j, c in enumerate([0, 0.25, 0.5, 0.75, 1]):
            dfoc = df.loc[df['obstacle_ratio'] == o].loc[df['conglomeration_ratio'] == c]
            grid[i, j] = np.sum(dfoc['waits'])

    grid[0, 1] = grid[0, 2] = grid[0, 3] = grid[0, 4] = grid[0, 0]

    # Create a heatmap
    sns.set()
    a = sns.heatmap(grid, annot=True, xticklabels=['0%', '25%', '50%', '75%', '100%'],
                    yticklabels=['0%', '25%', '50%', '75%', '90%'],
                    fmt=".0f", cmap='BuPu')
    a.invert_yaxis()

    # Add title
    plt.title("Mosse di attesa fatte dagli entry agent")
    plt.xlabel('Conglomerazione')
    plt.ylabel('Percentuale di ostacoli')
    plt.show()

    grid = np.zeros((4, len(df['obstacle_ratio'].unique())))
    for i, a in enumerate([0, 1, 2, 3]):
        for j, c in enumerate(df['obstacle_ratio'].unique()):
            dfoc = df.loc[df['num_agents'] == a].loc[df['obstacle_ratio'] == c]
            grid[i, j] = np.sum(dfoc['waits'])

    # Create a heatmap
    sns.set()
    a = sns.heatmap(grid, annot=True, fmt=".0f", cmap='BuPu', xticklabels=['0%', '25%', '50%', '75%', '90%'])
    a.invert_yaxis()

    # Add title
    plt.title("Numero di mosse di attesa fatte dagli entry agent")
    plt.xlabel('% di ostacoli')
    plt.ylabel('Numero di agenti')
    plt.show()
