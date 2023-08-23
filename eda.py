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
