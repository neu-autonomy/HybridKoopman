import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import itertools

def plot_mse(mse_dict, shift = 0, title=None):
    """
    Plot the Mean Squared Error (MSE) from a dictionary of MSE values.

    Parameters:
    mse_dict (dict): A dictionary where keys are labels (e.g., method names) and values are lists of MSE values.
    title (str): Title of the plot. Default is None.

    Returns:
    None: Displays the plot.
    """
    
    fig = plt.figure(figsize=(20, 14))  # Set the figure size for better visibility
    fontsize_base = max(fig.get_size_inches()) * 2 
    # colors are dark orange, blue, dark green,
    colors = ['#FF8C00', '#2694f9', '#2ca02c']  # Dark orange, blue, and dark green

    plotted_lines = {}
    for i, (label, mse_values) in enumerate(mse_dict.items()):
        default_linestyle = '-'  # Default line style
        # Check if the same exact curve has been plotted before
        mse_values = np.array(mse_values)  # Ensure mse_values is a numpy array for consistency
        for prev_label, prev_values in plotted_lines.items():
            if np.array_equal(mse_values, prev_values):
                print(f"Detected duplicate MSE values for label '{label}' and '{prev_label}'. Changing linestyle to dashed.")
                # If the current MSE values match a previously plotted line, change the linestyle
                default_linestyle = 'dotted'
                break

        # Store the current MSE values for comparison with future lines
        plotted_lines[label] = mse_values

        # NOTE: the data is collected every 5 epochs, so the x-axis will be multiplied by 5 to match the epochs
        x_epochs = np.arange(len(mse_values)) * 5 + shift  # Assuming each entry in mse_values corresponds to an epoch, adjust as needed
        plt.plot(x_epochs, mse_values, label=label, color=colors[i % len(colors)], linestyle=default_linestyle, linewidth=fontsize_base *0.1)
    
    plt.xlabel('Epochs', fontsize=fontsize_base * 0.8)  # Set x-axis label with larger font size
    plt.ylabel('Mean Squared Error (MSE)', fontsize=fontsize_base * 0.8)  # Set y-axis label with larger font size
    plt.yscale('log')  # Log scale for better visualization
    ax = plt.gca()  # Get current axis
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # place in top right outside the plot area
    plt.legend(loc='upper right', fontsize=fontsize_base * 0.7)  # Adjust legend location and font size
    # plt.tight_layout()  # Adjust layout to make room for the legend
    plt.tick_params(axis='both', which='major', labelsize=fontsize_base * 0.7)  # Set tick parameters for both axes
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.title(title if title else 'Mean Squared Error over Epochs', fontsize=fontsize_base * 1.2)  # Set title with default if None provided
    # legend is the names of the methods
    plt.show()

if __name__ == "__main__":

    experiments = [
        "DiscreteSpectrumExample",
        "FluidFlowOnAttractor",
        "Pendulum",
        "Lorenz"
    ]

    experiment_names = ["Discrete Spectrum",
                        "Fluid Flow On Attractor",
                        "Pendulum",
                        "Lorenz"
                        ]  # For display purposes in the plot title

    for experiment in experiments:

        # Example usage
        mse_data = {
            'Method A': [0.1, 0.09, 0.08, 0.07, 0.06],
            'Method B': [0.15, 0.14, 0.12, 0.11, 0.1],
            'Method C': [0.2, 0.18, 0.16, 0.15, 0.14]
        }

        root = "../models/final models/"
        # DISCRETESPECTRUMEXAMPLE
        if experiment == "DiscreteSpectrumExample":
            model_paths = [
                "DiscreteSpectrumExample_lusch_paramselated-donkey-159_nc0_nr2_mem1_loss.txt",
                "DiscreteSpectrumExample_no_hot_startdifferent-snow-158_nc0_nr2_mem1_loss.txt",
                "DiscreteSpectrumExample_hot_startvague-glitter-160_nc0_nr2_mem1_loss.txt",
            ]

        # FLUIDFLOWONATTRACTOR
        elif experiment == "FluidFlowOnAttractor":
            model_paths = [
                "FluidFlowOnAttractor_lusch_paramsfloral-eon-150_nc1_nr0_mem1_loss.txt",
                "FluidFlowOnAttractor_no_hot_startlilac-plant-153_nc1_nr1_mem1_loss.txt",
                "FluidFlowOnAttractor_hot_startgrateful-water-141_nc1_nr1_mem1_loss.txt",
            ]
        # PENDULUM
        elif experiment == "Pendulum":
            model_paths = [
                "Pendulum_lusch_paramswinter-breeze-155_nc1_nr0_mem1_loss.txt",
                "Pendulum_no_hot_startblooming-gorge-154_nc1_nr0_mem1_loss.txt",
                "Pendulum_hot_startrobust-eon-142_nc1_nr0_mem1_loss.txt",
            ]
        # LORENZ
        elif experiment == "Lorenz":
            model_paths = [
                "Lorenz_lusch_paramsdenim-pyramid-231_nc2_nr2_mem1_loss.txt",
                "Lorenz_no_hot_startdistinctive-sun-232_nc2_nr2_mem2_loss.txt",
                "Lorenz_hot_startgallant-gorge-233_nc2_nr2_mem2_loss.txt"
            ]
        else:
            raise ValueError("Experiment not recognized. Please provide a valid experiment name.")
        

        # Read the MSE values from the model paths
        model_names = ["Lusch Parameters", "Without Pretraining", "With Pretraining"]  # NOTE: keep this order consistent with the pathing!!
        mse_data = {}
        for i, model_path in enumerate(model_paths):
            mse_values = []
            
            try:
                with open(root + model_path, 'r') as f:
                    for line in f:
                        # if the line is all 0s, we can skip it
                        if int(line[0].strip()) == 0:
                            continue
                        # Assuming each line contains a single MSE value
                        mse_values.append(float(line.split(" ")[0].strip()))
            except FileNotFoundError:
                print(f"File {model_path} not found. Skipping.")
                continue
            
            mse_data[model_names[i]] = mse_values


        shift = 100

        if experiment == "Lorenz":
            shift = 500

        plot_mse(mse_data, shift, title='Test MSE on {} Dataset'.format(experiment_names[experiments.index(experiment)]))