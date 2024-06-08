####
# title: drift_gym.py
#
# language: python3
# author: Elmar Bucher
# date: 2024-05-07
# license: BSD 3-Clause
#
# run:
#     python3 test/test_episode.py
#
# description:
#     unit test code for the physigym project
#     note: pytest and physigym enviroment are incompatible.
#####


# modules
import glob
import gymnasium
import matplotlib.pyplot as plt
import os
import pandas as pd
import pcdl
import physigym
import random
import shutil


#############
# run tests #
#############
print('\nUNITTEST check for drift gym ...')


####################################
# run to generate data frame files #
####################################

# load PhysiCell Gymnasium environment
env = gymnasium.make(
    'physigym/ModelPhysiCellEnv-v0',
    #settingxml='config/PhysiCell_settings.xml',
    #render_mode='rgb_array',
    #render_fps=10
)

# reset output
os.system('rm timeseries_*_episode*.*')
os.system('rm -r output*')
os.system('make data-cleanup')

# episode loop
for i_episode in range(3):

    # reset variable
    random.seed(0)

    # reset the environment
    i_observation, d_info = env.reset(seed=0)
    r_reward = 0
    
    # episode time step loop
    b_episode_over = False
    while not b_episode_over:
        # policy fix or radom
	#r_dose = random.random()
        r_dose = 0.01
        d_action = {'drug_dose': r_dose}

        # action
        o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(d_action)

        # check if episode finsih
        b_episode_over = b_terminated or b_truncated
        print(f'dt_gym env step: {env.unwrapped.step_env}\tepisode: {env.unwrapped.episode}\tepisode step: {env.unwrapped.step_episode}\tover: {b_episode_over}\tb_terminated: {b_terminated}\tb_truncated: {b_truncated}')

    # save output
    shutil.move('output', f'output{str(i_episode).zfill(3)}')
    os.mkdir('output/')
    if (i_episode > 0):
        os.system(f'cp output000/initial_mesh0.mat output{str(i_episode).zfill(3)}')   # have to be fixed inthe source code!

# free PhysiCell Gymnasium environment
env.close()


#######################
# get timeseries data #
#######################
for s_path in glob.glob('output0*'):
    print(f'processing: {s_path}')
    i_episode = int(s_path.replace('output',''))
    mcdsts = pcdl.TimeSeries(s_path)
    df_cell = mcdsts.get_cell_df().drop({'runtime'}, axis=1)  # 'ID'
    df_conc = mcdsts.get_conc_df().drop({'runtime'}, axis=1)
    df_cell.to_csv(f'timeseries_cell_episode{str(i_episode).zfill(3)}.csv')
    df_conc.to_csv(f'timeseries_conc_episode{str(i_episode).zfill(3)}.csv')

    # plot timeseries
    fig, axs = plt.subplots(nrows=3, ncols=1 ,figsize=(8,12))
    mcdsts.plot_timeseries('cell_type', ax=axs[0])
    mcdsts.plot_timeseries('cell_type', 'drug', ax=axs[1])
    mcdsts.plot_timeseries('cell_type', 'death_rates_0', ax=axs[2])
    fig.suptitle(f'timeseries episode {str(i_episode).zfill(3)}')
    plt.tight_layout()
    fig.savefig(f'timeseries_plot_episode{str(i_episode).zfill(3)}.png')
print()

##################
# load conc data #
##################
ddf_conc = {}
for s_file in sorted(glob.glob('timeseries_conc_episode*csv')):
    i_episode = int(s_file.replace('timeseries_conc_episode','').replace('.csv',''))
    df_conc = pd.read_csv(s_file, index_col=0)
    ddf_conc.update({i_episode: df_conc})

# check results
for i_episode in ddf_conc.keys():
    print(f'processing: conc episode {i_episode}')
    for s_column in  ddf_conc[0].columns:
        if any(ddf_conc[0][s_column] != ddf_conc[i_episode][s_column]):
            print(f'\tepisode conc: {i_episode}\tcolumn: {s_column}')
print()

##################
# load cell data #
##################
ddf_cell = {}
for s_file in sorted(glob.glob('timeseries_cell_episode*csv')):
    i_episode = int(s_file.replace('timeseries_cell_episode','').replace('.csv',''))
    df_cell = pd.read_csv(s_file, index_col=0)
    ddf_cell.update({i_episode: df_cell})

# check results
for i_episode in ddf_cell.keys():
    print(f'processing: cell episode {i_episode}')
    try:
        for s_column in  ddf_cell[0].columns:
            if any(ddf_cell[0][s_column] != ddf_cell[i_episode][s_column]):
                print(f'\tcell episode: {i_episode}\tcolumn: {s_column}')
    except ValueError:
        print(f'\tcell episode: {i_episode}\t{ddf_cell[0].shape} {ddf_cell[i_episode].shape}\terror: series not identically labeled')
print()

# finish
print('UNITTEST: ok!')

