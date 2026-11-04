import importlib
from matplotlib.ticker import LogLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import helperfns
from utils import *
from torch.utils.data import DataLoader
from training_new import Lusch_Model


def plot_xyz_traj(og_traj, pred_trajs, title="Predicted Trajectories"):
    """
    Plot 3 plots for x y z components of the predicted trajectories for each model
    """

    colors = ['#FF8C00', '#2694f9', '#2ca02c']  # Dark orange, blue, and dark green

    labels = []
    handles = []
    # 2D or 3D plot based on the shape of the trajectory
    if og_traj.shape[1] != 3:
        fig, axs = plt.subplots(2, 1, figsize=(20, 12))
        fontsize_base = max(fig.get_size_inches()) * 2 
        fig.suptitle(title, fontsize=fontsize_base * 1.2)
    else:
        fig, axs = plt.subplots(3, 1, figsize=(20, 16))
        fontsize_base = max(fig.get_size_inches()) * 2 
        fig.suptitle(title, fontsize=fontsize_base * 1.2)

    # Plot original trajectory
    labels.append('Ground Truth')
    gt_pred_x = axs[0].plot(og_traj[:, 0], color='black', linewidth=fontsize_base * 0.2, marker='o', markersize = fontsize_base *0.5, markevery = 20)  # Plot the X component of the original trajectory
    axs[1].plot(og_traj[:, 1], color='black', linewidth=fontsize_base * 0.2, marker='o', markersize = fontsize_base * 0.6, markevery = 20)  # Plot the Y component of the original trajectory

    handles.extend(gt_pred_x)  # Add the line objects to handles for legend. only do x because only need 1 handle per 3 components

    if og_traj.shape[1] == 3:  # Check if Z component exists
        axs[2].plot(og_traj[:, 2], label='Original Z', color='black', linewidth=fontsize_base * 0.2, marker='o', markersize = fontsize_base *0.6, markevery = 20)  # Plot the Z component of the original trajectory

    # Plot predicted trajectories
    plotted_lines = {}

    for i, (label, pred_traj) in enumerate(pred_trajs.items()):
        labels.append(label)  # Add label for the legend
        default_linestyle = '--'
        # Check if the same exact curve has been plotted before
        for prev_label, prev_values in plotted_lines.items():
            if np.array_equal(pred_traj, prev_values):
                print(f"Detected duplicate predicted trajectory for label '{label}' and '{prev_label}'. Changing linestyle to dashed.")
                # If the current predicted trajectory matches a previously plotted line, change the linestyle
                # slightly shift the values so theyre visible               
                default_linestyle = "dotted"
                break
        # Store the current predicted trajectory for comparison with future lines
        plotted_lines[label] = pred_traj  # Store the current predicted trajectory for comparison with future lines

        # Plot each predicted trajectory
        line_pred_x = axs[0].plot(pred_traj[:,  0], linestyle = default_linestyle, color = colors[i], linewidth=fontsize_base * 0.2)  # Plot the X component of the predicted trajectory
        line_pred_y = axs[1].plot(pred_traj[:, 1], linestyle=default_linestyle, color = colors[i], linewidth=fontsize_base * 0.2)  # Plot the Y component of the predicted trajectory

        handles.extend(line_pred_x)  # Add the line objects to handles for legend. only do x because only need 1 handle per 3 components

        if og_traj.shape[1] == 3:
            line_pred_z = axs[2].plot(pred_traj[:,  2], linestyle='--', color=colors[i], linewidth=fontsize_base * 0.2) # Plot the Z component of the predicted trajectory
            # Add legend for Z component if it exists

    # Ensure legends are updated after all plots
    for i, ax in enumerate(axs):
        # Set labels for each subplot, only last one gets the xlabel
        if i == len(axs) - 1:
            ax.set_xlabel('Time Step', fontsize=fontsize_base * 0.8)  # Set x-axis label with larger font size
        # ax.set_yscale('symlog')  # Set y-axis scale to linear for all plots
        ax.grid(True)  # Add grid for better readability

    axs[0].set_ylabel('X Component', fontsize=fontsize_base * 0.8)  # Set y-axis label for the first subplot
    axs[1].set_ylabel('Y Component', fontsize=fontsize_base * 0.8)  # Set y-axis label for the second subplot
    if og_traj.shape[1] == 3:
        axs[2].set_ylabel('Z Component', fontsize=fontsize_base * 0.8)

    fig.legend(
        handles,  # Handles for the legend
        labels,   # Labels for each line in the legend
        loc='upper right',  # Location of the legend
        bbox_to_anchor=(0.98, 0.9),  # Adjust position to avoid overlap with plots
        fontsize = fontsize_base * 0.7  # Set legend font size
    )
    for ax in axs:
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))  # Set x-axis ticks to every 5 steps for better readability
        ax.tick_params(axis='both', which='major', labelsize=fontsize_base)  # Set tick parameters for both axes
    
    plt.subplots_adjust(right=0.7, hspace=0.35)
    plt.show()  # Show the plot

def plot_mse(og_traj, pred_trajs, title="mse of Predicted Trajectories"):
    """
    Plot the mse of predicted trajectories against the original trajectory.
    """
    mse_values = {}
    colors = ['#FF8C00', '#2694f9', '#2ca02c']  # Dark orange, blue, and dark green
    
    # Calculate mse for each predicted trajectory
    for label, pred_traj in pred_trajs.items():
        # Calculate mse between original and predicted trajectory
        mse = np.mean((og_traj - pred_traj) ** 2, axis=1)
        mse_values[label] = mse

    plotted_lines = {}
    # Plotting mse values
    fig = plt.figure(figsize=(12, 12))  # Create a new figure for mse plot
    fontsize_base = max(fig.get_size_inches()) * 2 
    # Check for duplicate mse values and adjust linestyle if necessary
    for i, (label, mse) in enumerate(mse_values.items()):
        default_linestyle = '-'
        # Check if the same exact mse values have been plotted before
        for prev_label, prev_values in plotted_lines.items():
            if np.array_equal(mse, prev_values):
                print(f"Detected duplicate mse values for label '{label}' and '{prev_label}'. Changing linestyle to dashed.")
                # If the current mse values match a previously plotted line, change the linestyle
                default_linestyle = 'dotted'
                break
        # Store the current mse values for comparison with future lines
        plotted_lines[label] = mse

        plt.plot(mse, label=f'{label} MSE', color=colors[i], linestyle=default_linestyle, linewidth=fontsize_base*0.2)  # Plot mse for each predicted trajectory

    plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=10))
    plt.xlabel('Time Step', fontsize=fontsize_base )  # Set x-axis label with larger font size
    plt.yscale('log')  # Set y-axis scale to linear for mse plot
    plt.ylabel('L2 Error', fontsize=fontsize_base )  # Set y-axis label with larger font size
    # plt.title(title, fontsize=fontsize_base * 1.2)  # Set title with larger font size
    plt.tick_params(axis='both', which='both', labelsize=fontsize_base * 1.2)  # Set tick parameters for both axes
    plt.tick_params(axis='y', which='both', length=0)  # Hides tick marks
    plt.yticks([1e-3, 1e-2], labels=["$10^{-3}$", "$10^{-2}$"])  # Show only limits

    # plt.legend(
    #     fontsize = fontsize_base * 0.7,  # Set legend font size
    #     loc='best',  # Location of the legend
    #     # bbox_to_anchor=(1.4, 1),  # Adjust position to avoid overlap with plots
    # )
    # plt.grid(True)
    plt.show()  # Show the plot
    

def get_trajs(model, batch, val_loader, load):
    encoder = model.encoder.cpu()
    decoder = model.decoder.cpu()
    auxillary = model.auxillary
    for aux_model in auxillary:
        aux_model.cpu()

    checkpoint = torch.load(load, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    for i, aux_model in enumerate(auxillary):
        aux_model.load_state_dict(checkpoint['auxillary_state_dict'][i])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    traj= val_loader.dataset.reconstruct_one_batch(batch) 

    og_traj = traj
    pred_traj, _ = n_step_predict_vec(model, params, og_traj, n_steps = 1, use_log=False)
    return og_traj.cpu().detach().numpy() , pred_traj.cpu().detach().numpy()  # Return original and predicted trajectory

if __name__ == "__main__":

    # Example usage
    # Original trajectory (for example purposes)
    og_traj = np.array([[i, 2*i, 3*i] for i in range(51)])  # Example original trajectory
    # Predicted trajectories from different models
    pred_trajs = [
        ('Model 1', np.array([[i + np.random.normal(0, 0.1), 2*(i + np.random.normal(0, 0.1)), 3*(i + np.random.normal(0, 0.1))] for i in range(51)])),  # Model 1 predictions
        ('Model 2', np.array([[i + np.random.normal(0, 0.2), 2*(i + np.random.normal(0, 0.2)), 3*(i + np.random.normal(0, 0.2))] for i in range(51)]))   # Model 2 predictions
    ]

    
    exp_names = ["DiscreteSpectrumExample", "FluidFlowOnAttractor", "Pendulum", "Lorenz"]  # List of experiment names
    # exp_names = ["FluidFlowOnAttractor"]  # List of experiment names to plot trajectories for
    data_subset = {"train":"train1_x", "test":"test_x"}
    # wandb_enabled =False 

    for exp_name in exp_names:

        # DISCRETESPECTRUMEXAMPLE
        if exp_name == "DiscreteSpectrumExample":
            model_paths = [
                "DiscreteSpectrumExample_lusch_paramselated-donkey-159_nc0_nr2_mem1_epoch_999.pth",
                "DiscreteSpectrumExample_no_hot_startdifferent-snow-158_nc0_nr2_mem1_epoch_999.pth",
                "DiscreteSpectrumExample_hot_startvague-glitter-160_nc0_nr2_mem1_epoch_999.pth",
            ]

        # FLUIDFLOWONATTRACTOR
        elif exp_name == "FluidFlowOnAttractor":
            model_paths = [
                "FluidFlowOnAttractor_lusch_paramsfloral-eon-150_nc1_nr0_mem1_epoch_999.pth",
                "FluidFlowOnAttractor_no_hot_startlilac-plant-153_nc1_nr1_mem1_epoch_999.pth",
                "FluidFlowOnAttractor_hot_startgrateful-water-141_nc1_nr1_mem1_epoch_999.pth",
            ]
        # PENDULUM
        elif exp_name == "Pendulum":
            model_paths = [
                "Pendulum_lusch_paramswinter-breeze-155_nc1_nr0_mem1_epoch_999.pth",
                "Pendulum_no_hot_startblooming-gorge-154_nc1_nr0_mem1_epoch_999.pth",
                "Pendulum_hot_startrobust-eon-142_nc1_nr0_mem1_epoch_999.pth",
            ]
        # LORENZ
        elif exp_name == "Lorenz":
            model_paths = [
                "Lorenz_lusch_paramsdenim-pyramid-231_nc2_nr2_mem1_epoch_2999.pth",  
                "Lorenz_no_hot_startdistinctive-sun-232_nc2_nr2_mem2_epoch_2999.pth",  
                "Lorenz_hot_startgallant-gorge-233_nc2_nr2_mem2_epoch_2999.pth",  
            ]
        else:
            raise ValueError("Experiment not recognized. Please provide a valid experiment name.")

        params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
        helperfns.set_defaults(params)
        learning_rate = params['learning_rate']
        batch_size = 1
        traj_len = params['len_time']
        params['num_shifts_middle'] = 6
        k = params['num_shifts_middle']
        # traj_len = params['num_shifts']
        # k = params['num_shifts_middle']
        lambdas = {
            "recon": params['recon_lam'],
            "forward": params['recon_lam'],
            "linearity": params['mid_shift_lam'],
            "L1": params["L1_lam"],
            "L2": params["L2_lam"],
            "Linf": params["Linf_lam"]
        }

        # # Load precomputed koopman operatoTr
        koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables") # TODO remove fake
        koopman, d, omegas  = load_koopman(koopman_path)
        koopman = koopman.to(device)
        
        hidden_size = koopman.shape[0]
        
        d = 1
        train_dataset = pd.read_csv('../../data/csv/final_csv/{}_{}.csv'.format(exp_name, data_subset["train"]), delimiter=',', header=None, dtype=np.float64).values

        train_min = np.min(train_dataset, axis=0)
        train_max = np.max(train_dataset, axis=0)

        test_dataset = pd.read_csv('../../data/csv/final_csv/{}_{}.csv'.format(exp_name, data_subset["test"]), delimiter=',', header=None, dtype=np.float64).values
        test_dataset = 2.0 * (test_dataset - train_min) / (train_max - train_min) - 1.0
        test_dataset = KoopData(test_dataset, traj_len=traj_len, k=0, d=d)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


        # Define model
        # params['num_complex_pairs'] = 1
        # params['num_real'] = 2
        if exp_name == "Lorenz":
            d=2

        og_traj = None
        pred_trajs= {}
        experiments = ["Discrete Spectrum", "Fluid Flow On Attractor", "Pendulum", "Lorenz"]
        # experiments = ["Fluid Flow On Attractor"]  # List of experiment names for plotting
        # experiments = ["Lorenz"]  # Only Lorenz
        labels = ["Lusch Parameters", "W/o Pretraining", "W/ Pretraining"]  # Labels for the models OTHER THAN LORENZ
        # For Lorenz, we will use dif labels
        if exp_name == "Lorenz":
            labels = ["Parameter Sweep", "W/o Pretraining", "W/ Pretraining"]  # Labels for the models

        koopman_modes = ["lusch_params", "no_hot_start", "hot_start"]  # Different modes for the Koopman model
        # get random batch from the test_loader
        # random_b_start = np.random.randint(0, len(test_loader))  # Get a random starting index for the batch
        first_batch = next(iter(test_loader))  # Get the first batch from the test_loader to ensure we have data to work with
        # Get all batches as a list
        batches = list(test_loader)

        # Pick a random batch index
        random_b_start = np.random.randint(0, len(batches) - 1)

        # Get the batch by index
        random_batch = batches[random_b_start]
        random_batch = first_batch
        # Loop through each model path and get trajectories
        for i, model_path in enumerate(model_paths):
            # Construct full path to the model checkpoint
            ckpt = f"../models/final models/{model_path}"
            model = Lusch_Model(params, koopman_mode=koopman_modes[i], koopman=koopman, omegas=omegas, d=d).to(device)

            # Get original and predicted trajectories
            original_traj, predicted_traj = get_trajs(model, random_batch, test_loader, load=ckpt)

            # Store the original trajectory
            if og_traj is None:
                og_traj = original_traj.squeeze(0)  # Assuming the first batch is representative of the original trajectory
            # Store the predicted trajectory with a label
            pred_trajs[labels[i]] = predicted_traj.squeeze(0)  # Assuming the first batch is representative of the predicted trajectory

        # Now plot the original and predicted trajectories
        plot_xyz_traj(
            og_traj, pred_trajs, title="Predicted Trajectories in {}".format(experiments[exp_names.index(exp_name)])  # Use the experiment name to title the plot
        )    

        plot_mse(
            og_traj, pred_trajs, title="mse in {}".format(experiments[exp_names.index(exp_name)] + " Dataset")  # Use the experiment name to title the plot
        )