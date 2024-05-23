import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from yolo_tools.database.FlyChoiceDatabase import DatabaseHandler


def boxplot_by_genotype(df,quantification = 'prefrence_index', experiment_title= '', notch = True):
    """
    Plots the preference index by genotype with sex as the hue using seaborn boxplots.

    Args:
        df (pd.DataFrame): The DataFrame containing the experimental data.
    """
    # Set up the color palette
    palette = {True: "orange", False: "blue"}
    
    # Create the plot
    fig= plt.figure(figsize=(12, 8))
    boxplot = sns.boxplot(x='genotype_shortname', y= quantification, hue='fly_is_female', data=df, palette=palette,notch=notch)
    
    # Customize the plot
    plt.title(f'{quantification.replace('_',' ')} | {experiment_title}')
    plt.xlabel('Genotype')
    plt.ylabel(quantification.replace('_',' '))
    plt.xticks(rotation=45)
    
    # Set the legend
    handles, labels = boxplot.get_legend_handles_labels()
    boxplot.legend(handles, ['Female', 'Male'], title='Sex')

    
    return fig
# Plotting function
def add_arena(ax,stim_left,stim_right):
    """
    Adds a rectangle, a vertical line, and text annotations to the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to add the rectangle, line, and text to.
    """
    # Plot the rectangle (18 mm wide and 8 mm high)
    rect = plt.Rectangle((0, 0), 18, 8, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Plot the vertical line from (9, 0) to (9, 8)
    ax.plot([9, 9], [0, 8], color='black', linewidth=1)
    
    # Add the left-justified text "stim1" at position (0, -1)
    ax.text(0, -1, stim_left, ha='left')
    
    # Add the right-justified text "stim2" at position (18, -1)
    ax.text(18, -1, stim_right, ha='right')

    # Setting the limits to ensure the rectangle and text are visible
    ax.set_xlim(-1, 19)
    ax.set_ylim(-2, 9)

    # Equal aspect ratio
    ax.set_aspect('equal')


def plot_trajectory(df, fps,stim_left,stim_right):
    """
    Plots the trajectory from the DataFrame with equally spaced axes and viridis colormap.
    The colorbar represents time in seconds.

    Args:
        df (pd.DataFrame): The DataFrame containing the trajectory data.
        fps (float): The frames per second to convert index to time in seconds.
        
    Returns:
        fig (matplotlib.figure.Figure): The figure handle.
    """
    x = df['pos_x_mm_arena_centered']
    y = df['pos_y_mm_arena_centered']
    time_in_seconds = df.index / fps
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots()
    norm = plt.Normalize(time_in_seconds.min(), time_in_seconds.max())
    lc = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    lc.set_array(time_in_seconds)

    for i in range(len(segments)):
        seg = segments[i]
        ax.plot(seg[:, 0], seg[:, 1], color=plt.cm.viridis(norm(time_in_seconds[i])), linewidth=2)

    ax.set_aspect('equal')
    ax.set_title('Trajectory Plot')
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    plt.colorbar(lc, ax=ax, label='Time (seconds)')

    # Add the rectangle and vertical line
    add_arena(ax,stim_left,stim_right)
    return fig

def create_identifier(df):
    df['identifier'] = (
        df['stimulus_01_name'].astype(str) + ' ' +
        df['stimulus_01_amplitude'].astype(str) + df['stimulus_01_amplitude_unit'].astype(str) +
        ' vs. ' +
        df['stimulus_02_name'].astype(str) + ' ' +
        df['stimulus_02_amplitude'].astype(str) + df['stimulus_02_amplitude_unit'].astype(str)
    )
    return df


# database url
db_filepath = '/home/geuba03p/fly_choice.db'
db_handler = DatabaseHandler(f'sqlite:///{db_filepath}')

df = db_handler.get_two_choice_results()
not_far_enough = df.distance_walked_mm.isna() | (df.distance_walked_mm <100)
df = df.loc[not_far_enough == False,:]
df = create_identifier(df)

figure_list = list()
for experiment_identifier in df.identifier.unique():

    df_subset = df.loc[df.identifier == experiment_identifier]


    quants = ['distance_walked_mm', 'max_speed_mmPs', 'avg_speed_mmPs', 'fraction_left', 'fraction_right',
            'fraction_middle', 'fraction_positive', 'fraction_negative',
            'preference_index', 'decision_to_positive_num',
            'decision_from_positive_num', 'decision_to_negative_num',
            'decision_from_negative_num', 'duration_after_positive',
            'duration_after_negative']


    quants = ['distance_walked_mm', 'max_speed_mmPs', 'avg_speed_mmPs', 
            'fraction_middle', 'fraction_positive', 'fraction_negative',
            'preference_index']


    # for q in quants:
    #     fig = boxplot_by_genotype(df_subset,q,experiment_identifier)
    #     fig_name =f'{experiment_identifier}_{q}'
    #     figure_list.append((fig,fig_name))


# Example usage
tid = 154
tra_df =db_handler.get_trajectory_for_trial(tid)
row = df.loc[df.trial_id ==tid,:]
stim_left = f'{row.stimulus_01_name.iloc[0]} {row.stimulus_01_amplitude.iloc[0]} {row.stimulus_01_amplitude_unit.iloc[0]}'
stim_right = f'{row.stimulus_02_name.iloc[0]} {row.stimulus_02_amplitude.iloc[0]} {row.stimulus_02_amplitude_unit.iloc[0]}'
fig = plot_trajectory(tra_df,df.experiment_fps[0], stim_left, stim_right)
figure_list.append((fig,'example_trajectory'))

save_dir =  '/home/geuba03p/koen_figs'
for fig_data in figure_list:
    fig, fig_name = fig_data
    for ext in ['svg','png']:
        fig.savefig(f'{save_dir}/{fig_name}.{ext}')


plt.show()

print('wait')