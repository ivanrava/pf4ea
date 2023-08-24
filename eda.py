import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    sns.violinplot(x='size', y='memory', data=df)
    plt.show()




    # graphic size - resolution time (filtered on status success)
    df_success = df.loc[df['status'] == 'success']

    df_diagonal = df_success.loc[df['h'] == 'diagonal']
    df_dijkstra = df_success.loc[df['h'] == 'dijkstra']

    #grafico size - process time (filtered on success )
    sns.lineplot(data=df_diagonal.groupby('size').agg(ProcessTimeMeanDiagonal=('process_time','mean')),  palette=['red'])
    sns.lineplot(data=df_dijkstra.groupby('size').agg(ProcessTimeMeanDijkstra=('process_time', 'mean')), palette=['blue'])
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
