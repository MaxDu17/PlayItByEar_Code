#this file looks into the communal results direcotry and generates the appropiate statistics and plots.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "ADD DIRECTORY HERE"

os.chdir(base_dir)

#this function will go through the communal_results directory and concate all relevant logs into one dataframe. It retuns the mean and sample STD
def grab_and_concatenate_logs(modifier = "", sample_size = 1, seed="all"):
    indicator_list = list()
    blocked_list = list()

    #modifier is used for the on-policy corrections and balanced batches
    index = None

    for directory in sorted(os.listdir()):
        if "IndicatorBoxBlock" in directory and (seed == "all" or str(seed) in directory.split("_")[2]):
            try:
                df = pd.read_csv(directory + "/" + modifier + "eval.csv")

                indicator_list.append(df)
                index = df['step']
                print(f"\tSuccessfully parsed {directory}/{modifier}eval.csv")
            except FileNotFoundError:
                print(f"The directory {directory} does not have a valid eval file")
            except pd.errors.EmptyDataError:
                print(f"The directory {directory} has a blank file")
        elif "BlockedPickPlace" in directory and (seed == "all" or str(seed) in directory.split("_")[2]):
            try:
                df = pd.read_csv(directory + "/" + modifier + "eval.csv")
                blocked_list.append(df)
                print(f"\tSuccessfully parsed {directory}/{modifier}eval.csv")
            except FileNotFoundError:
                print(f"The directory {directory} does not have a valid eval file")
            except pd.errors.EmptyDataError:
                print(f"The directory {directory} has a blank file")
        else:
            print(f"Skipped {directory}")
    try:
        indicator_concat = pd.concat(indicator_list).drop('average_episode_reward', axis = 'columns')
        blocked_concat = pd.concat(blocked_list).drop('average_episode_reward', axis = 'columns')
    except: #this is for the corrections
        indicator_concat = pd.concat(indicator_list)
        blocked_concat = pd.concat(blocked_list)

    #not sure if this is the right statistical method but using it for now
    return indicator_concat.groupby('step').mean().to_numpy()[:, 0], indicator_concat.groupby('step').std().to_numpy()[:, 0] / np.sqrt(len(indicator_list) * sample_size), blocked_concat.groupby('step').mean().to_numpy()[:, 0],blocked_concat.groupby('step').std().to_numpy()[:, 0] / np.sqrt(len(indicator_list) * sample_size), index


#plotting raw BC training curves. Not used at the moment
def plot_BC_progress():
    plt.style.use("seaborn")
    plt.grid()

    i_mean, i_std, b_mean, b_std, rows = grab_and_concatenate_logs(modifier = "resid/")
    rows = rows.to_numpy()


    fig, (ax1, ax2) = plt.subplots(nrows = 2)

    plt.title("BC Training Progress")
    ax1.plot(rows, i_mean, color = 'orange')
    ax1.fill_between(rows, i_mean - i_std, i_mean + i_std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.set_ylabel("Suceess Proportion")
#     ax1.set_xlabel("BC Train Steps")
    ax1.set_title("IndicatorBoxBlock")

    ax2.plot(rows, b_mean, color = 'orange')
    ax2.fill_between(rows, b_mean - b_std, b_mean + b_std, alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
        linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.set_ylabel("Suceess Proportion")
    ax2.set_xlabel("BC Train Steps")
    ax2.set_title("BlockedPickPlace")
#     ax.grid()
    plt.tight_layout()
    fig.savefig("BC_training_progress.png")


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.asarray(smoothed)


# generates two separate plots for the two environments
def plot_correction_progress(seed):
    smooth_factor = 0
    modifier = "corrections_improved"
    plt.style.use("seaborn")
    plt.grid()

    #modifier is the subdirectory to use under the main directory.
    i_mean_audio, i_std_audio, b_mean_audio, b_std_audio, _ = grab_and_concatenate_logs(modifier = "ABLATE_MEM_EVAL/", sample_size = 1, seed=seed) #don't divide
    i_mean_BC, i_std_BC, b_mean_BC, b_std_BC, _ = grab_and_concatenate_logs(modifier = "100000/", sample_size = 1, seed=seed) #don't divide


    fig, ax1 = plt.subplots()
    i_mean, i_std, b_mean, b_std, rows = grab_and_concatenate_logs(modifier = f"{modifier}/", sample_size = 1, seed=seed)

    #to generate the horizontal line
    i_mean_BC = np.tile(i_mean_BC, len(rows))
    b_mean_BC = np.tile(b_mean_BC, len(rows))
    i_mean_audio = np.tile(i_mean_audio, len(rows))
    b_mean_audio = np.tile(b_mean_audio, len(rows))

    plt.title("Corrections Progress")
    ax1.plot(rows, i_mean_BC, color = '#FF8360')
    ax1.fill_between(rows, i_mean_BC - i_std_BC, i_mean_BC + i_std_BC, alpha=0.2, facecolor='#FF8360',
        linewidth=0, antialiased=True)

    ax1.plot(rows, i_mean_audio, color = '#0B2027')
    ax1.fill_between(rows, i_mean_audio - i_std_audio, i_mean_audio + i_std_audio, alpha=0.2, facecolor='#0B2027',
        linewidth=0, antialiased=True)


    ax1.plot(rows, smooth(i_mean, smooth_factor), color = '#658e9c')
    ax1.fill_between(rows, smooth(i_mean, smooth_factor) - i_std, smooth(i_mean, smooth_factor) + i_std, alpha=0.2, facecolor='#658e9c',
        linewidth=0, antialiased=True)

    ax1.set_ylabel("Suceess Proportion")
    ax1.set_xlabel("Intervention Episodes")
    ax1.set_title("IndicatorBoxBlock")
    ax1.set_ylim(0.2, 0.6)


    fig.savefig(f"corrections_plot_Indicator_{seed}_{modifier}.png")

    fig, ax1 = plt.subplots()

    plt.title("Corrections Progress")
    ax1.plot(rows, b_mean_BC, color = '#FF8360')
    ax1.fill_between(rows, b_mean_BC - b_std_BC, b_mean_BC + b_std_BC, alpha=0.2, facecolor='#FF8360',
        linewidth=0, antialiased=True)

    ax1.plot(rows, b_mean_audio, color = '#0B2027')
    ax1.fill_between(rows, b_mean_audio - b_std_audio, b_mean_audio + b_std_audio, alpha=0.2, facecolor='#0B2027',
        linewidth=0, antialiased=True)

    ax1.plot(rows, smooth(b_mean, smooth_factor), color = '#658e9c')
    ax1.fill_between(rows, smooth(b_mean, smooth_factor) - b_std, smooth(b_mean, smooth_factor)+ b_std, alpha=0.2, facecolor='#8cba80',
        linewidth=0, antialiased=True)

    ax1.set_ylabel("Suceess Proportion")
    ax1.set_xlabel("Intervention Episodes")
    ax1.set_title("BlockedPickPlace")
    ax1.set_ylim(0.2, 0.8)


    fig.savefig(f"corrections_plot_PickPlace_{seed}_{modifier}.png")

# plot_correction_progress("all")
plot_BC_progress()
