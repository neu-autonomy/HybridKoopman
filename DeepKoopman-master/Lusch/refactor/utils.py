from scipy.io import loadmat, savemat
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import wandb
from time import time
from torch.utils.data import Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class KoopData(Dataset):
    def __init__(self, data, traj_len, k, d):
        # pass
        # input 3D shifted data
        # output recreation_loss + Koopman_forward_loss + linearity_loss
        self.traj_len = traj_len
        self.k = k
        self.d = d
        self._raw_data = data 
        data = torch.tensor(data, dtype=torch.float32)
        self.dim = data.shape[-1]

        if data.shape[0] % traj_len != 0:
            # kill the end of the data until it is divisible by traj_len
            data = data[:data.shape[0] - (data.shape[0] % traj_len)]
        # reshape the data into trajectories
        data = data.reshape(data.shape[0] // traj_len, traj_len,  data.shape[1])

        # inflate state dimension in accordance to memory
        mem_data = torch.zeros((data.shape[0], traj_len-d+1,  data.shape[-1] * d))
        for i in range(traj_len - d + 1):
            inflated_state = data[:, i:i+d,:].reshape(data.shape[0], -1)
            mem_data[:,i] = inflated_state

        #extend in a new dim the k-length data after the current time
        # for each state in a given trajectory, concat the next k states in a new dim
        # rolled_data_k = torch.stack([torch.roll(data[:, :data.shape[1]-k], -i, dims=1) for i in range(1, k+1)], dim=3)

        rolled_data_k = torch.zeros(mem_data.shape[0], mem_data.shape[1]-k, mem_data.shape[2], k+1)
        for i in range(0, k+1):
            rolled_data_k[:, :, :, i] = mem_data[:, i:mem_data.shape[1]-k+i,:]
        self.data = rolled_data_k
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def reconstruct_one_batch(self, batch, k=None, d=None):
        if k is None:
            k = self.k
        if d is None:
            d = self.d
        # takes in a batch of [batch_size, traj_len-k-d+1, dim, k] and returns a traj of traj_len
        batch_size = batch.shape[0]
        data = torch.zeros(batch_size, self.traj_len, self.dim)
        for i in range(self.traj_len - k-d+1):
            data[:,i, :] = batch[:,i,0:self.dim,0]
        for j in range(0,k+1):
            data[:, self.traj_len-k-d+j:self.traj_len-k+j, :] = batch[:, -1, :, j].reshape(batch_size,d, self.dim)
        return data

    def reconstruct_one_batch_k(self, batch, n_steps):
        # takes in a batch of [batch_size, traj_len-k, dim, k] and returns a traj of traj_len
        data = self.reconstruct_one_batch(batch)
        # mem_data = torch.zeros((data.shape[0], self.traj_len-self.d+1,  data.shape[-1] * self.d))
        # mem_data = torch.zeros((data.shape[0], self.traj_len-n_steps,  data.shape[-1] * self.d))
        # for i in range(self.traj_len - n_steps+1):
        #     inflated_state = data[:, i:i+n_steps,:].reshape(data.shape[0], -1)
        #     mem_data[:,i] = inflated_state

        #extend in a new dim the k-length data after the current time
        # for each state in a given trajectory, concat the next k states in a new dim
        # rolled_data_k = torch.stack([torch.roll(data[:, :data.shape[1]-k], -i, dims=1) for i in range(1, k+1)], dim=3)

        rolled_data_k = torch.zeros(data.shape[0], self.traj_len-n_steps, data.shape[2], n_steps)
        for i in range(0, n_steps):
            rolled_data_k[:, :, :, i] = data[:, i+1:data.shape[1]-n_steps+i+1,:]

        return rolled_data_k

@torch.compile
def form_pkoop_from_auxillary(params, complex_omegas, real_omegas, delta_t):
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']
    batch_size = complex_omegas[0].shape[0] if num_complex_pairs > 0 else real_omegas[0].shape[0]

    # Process complex eigenvalues into 2×2 blocks

    if num_complex_pairs > 0:

        complex_omegas = torch.stack(complex_omegas, dim=1)
        real_part = complex_omegas[:, :, 0]  # Extract real part ################
        imag_part = complex_omegas[:, :, 1]  # Extract imaginary part ##################
        
        # scale = torch.exp(real_part * delta_t)  # Exponential decay/growth factor #####
        scale = real_part ** delta_t  # Exponential decay/growth factor #####
        cos_vals = torch.cos(imag_part * delta_t)
        sin_vals = torch.sin(imag_part * delta_t)

        row1 = torch.stack([scale * cos_vals, -scale * sin_vals], dim=-1)  # Shape: [batch_size, num_complex_pairs, 2]
        row2 = torch.stack([scale * sin_vals, scale * cos_vals], dim=-1)   # Shape: [batch_size, num_complex_pairs, 2]
        # complex_blocks = torch.stack([row1, row2], dim=-2)  # Shape: [batch_size, num_complex_pairs, 2, 2]
        
        if num_complex_pairs == 1:
            complex_blocks = torch.stack([row1, row2], dim=-2)  # Shape: [batch_size, num_complex_pairs, 2, 2]
            complex_blocks = complex_blocks.view(batch_size, 2 * num_complex_pairs, 2 * num_complex_pairs) #############################
        else:
            complex_blocks = torch.stack([row1, row2], dim=-2) 
            complex_components=[]
            for i in range(0,num_complex_pairs):
                complex_components.append(torch.cat((torch.zeros((batch_size, 2,2*i)).to(device), complex_blocks[:,i,:,:], torch.zeros((batch_size,2,2*num_complex_pairs-2*(i+1))).to(device)), dim=2))
                # complex_components.append(torch.cat((torch.zeros((batch_size, 2*i, 2)).to(device), complex_blocks[:,i,:,:], torch.zeros((batch_size,num_complex_pairs-2*i, 2)).to(device)), dim=1))           

            complex_blocks = torch.cat(complex_components, dim=1)

        #########################
        # Reshape to form block diagonal structure for complex eigenvalues
        # breakpoint()
        complex_blocks = complex_blocks.view(batch_size, 2 * num_complex_pairs, 2 * num_complex_pairs) #############################
    else:
        complex_blocks = None

    # Process real eigenvalues into diagonal form
    if num_real > 0:
        nn_omegas_real = torch.stack(real_omegas, dim=1)  # Shape: (batch_size, num_real)

        # real_blocks = torch.diag_embed(torch.exp(nn_omegas_real * delta_t))  # Shape: [batch_size, num_real, num_real]
        real_blocks = torch.diag_embed(nn_omegas_real ** delta_t)  # Shape: [batch_size, num_real, num_real]
        # if torch.any(torch.isnan(real_blocks)) or torch.any(torch.isinf(real_blocks)):
        #     breakpoint()
    else:
        real_blocks = None
    # Combine into block-diagonal Koopman operator
    if complex_blocks is not None and real_blocks is not None:
        koopman_matrices = torch.cat([
            complex_blocks,
            torch.zeros(batch_size, 2 * num_complex_pairs, num_real, device=complex_blocks.device)
        ], dim=-1)  # Append zero padding horizontally
        real_padding = torch.cat([
            torch.zeros(batch_size, num_real, 2 * num_complex_pairs, device=real_blocks.device),
            real_blocks
        ], dim=-1)  # Append real blocks with zero padding
        koopman_matrices = torch.cat([koopman_matrices, real_padding], dim=-2)  # Append vertically

    elif complex_blocks is not None:
        koopman_matrices = complex_blocks
    else:
        koopman_matrices = real_blocks

    return koopman_matrices

@torch.compile
def form_koopman_from_auxillary(params, complex_omegas, real_omegas, delta_t):
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']
    batch_size = complex_omegas[0].shape[0] if num_complex_pairs > 0 else real_omegas[0].shape[0]

    # Process complex eigenvalues into 2×2 blocks

    if num_complex_pairs > 0:

        complex_omegas = torch.stack(complex_omegas, dim=1)
        real_part = complex_omegas[:, :, 0]  # Extract real part ################
        imag_part = complex_omegas[:, :, 1]  # Extract imaginary part ##################
        
        scale = torch.exp(real_part * delta_t)  # Exponential decay/growth factor #####
        cos_vals = torch.cos(imag_part * delta_t)
        sin_vals = torch.sin(imag_part * delta_t)

        row1 = torch.stack([scale * cos_vals, -scale * sin_vals], dim=-1)  # Shape: [batch_size, num_complex_pairs, 2]
        row2 = torch.stack([scale * sin_vals, scale * cos_vals], dim=-1)   # Shape: [batch_size, num_complex_pairs, 2]
        # complex_blocks = torch.stack([row1, row2], dim=-2)  # Shape: [batch_size, num_complex_pairs, 2, 2]
        
        if num_complex_pairs == 1:
            complex_blocks = torch.stack([row1, row2], dim=-2)  # Shape: [batch_size, num_complex_pairs, 2, 2]
            complex_blocks = complex_blocks.view(batch_size, 2 * num_complex_pairs, 2 * num_complex_pairs) #############################
        else:
            complex_blocks = torch.stack([row1, row2], dim=-2) 
            complex_components=[]
            for i in range(0,num_complex_pairs):
                # complex_components.append(torch.cat((torch.zeros((batch_size, 2,2*i)).to(device), complex_blocks[:,i,:,:], torch.zeros((batch_size,2,2*num_complex_pairs-2*i)).to(device)), dim=2))
                complex_components.append(torch.cat((torch.zeros((batch_size, 2,2*i)).to(device), complex_blocks[:,i,:,:], torch.zeros((batch_size,2,2*num_complex_pairs-2*(i+1))).to(device)), dim=2))
                # complex_components.append(torch.cat((torch.zeros((batch_size, 2*i, 2)).to(device), complex_blocks[:,i,:,:], torch.zeros((batch_size,num_complex_pairs-2*i, 2)).to(device)), dim=1))           

            complex_blocks = torch.cat(complex_components, dim=1)

        #########################
        # Reshape to form block diagonal structure for complex eigenvalues
        complex_blocks = complex_blocks.view(batch_size, 2 * num_complex_pairs, 2 * num_complex_pairs) #############################
    else:
        complex_blocks = None

    # Process real eigenvalues into diagonal form
    if num_real > 0:
        nn_omegas_real = torch.stack(real_omegas, dim=1)  # Shape: (batch_size, num_real)

        real_blocks = torch.diag_embed(torch.exp(nn_omegas_real * delta_t))  # Shape: [batch_size, num_real, num_real]
        # if torch.any(torch.isnan(real_blocks)) or torch.any(torch.isinf(real_blocks)):
        #     breakpoint()
    else:
        real_blocks = None
    # Combine into block-diagonal Koopman operator
    if complex_blocks is not None and real_blocks is not None:
        koopman_matrices = torch.cat([
            complex_blocks,
            torch.zeros(batch_size, 2 * num_complex_pairs, num_real, device=complex_blocks.device)
        ], dim=-1)  # Append zero padding horizontally
        real_padding = torch.cat([
            torch.zeros(batch_size, num_real, 2 * num_complex_pairs, device=real_blocks.device),
            real_blocks
        ], dim=-1)  # Append real blocks with zero padding
        koopman_matrices = torch.cat([koopman_matrices, real_padding], dim=-2)  # Append vertically

    elif complex_blocks is not None:
        koopman_matrices = complex_blocks
    else:
        koopman_matrices = real_blocks

    return koopman_matrices

def get_omegas_from_auxillary(model, params, encoded_data):
    auxillary = model.auxillary
    for aux in auxillary:
        aux.to(device)
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']

    nn_omegas_complex = [] 
    nn_omegas_real = []

    for j in range(num_complex_pairs):
        ind = j * 2 
        radius = torch.linalg.norm(encoded_data[:, ind:ind+2].unsqueeze(1), dim = 2)
        nn_omegas_complex.append(auxillary[j](radius).squeeze(1)) # check whether this should be ind or ind + 2
    for j in range(num_real):
        ind = j + 2*num_complex_pairs
        nn_omegas_real.append(auxillary[j + num_complex_pairs](encoded_data[:, ind].unsqueeze(1)).squeeze(1))

    return nn_omegas_complex, nn_omegas_real

def load_koopman(path=None, use_log=False):
    if path is None:
        path = '/home/spirolife/KoopmanGrowth/DeepKoopman-master/sznaier/koopman_out.mat' #TODO Get rid of this
    data = loadmat(path)
    Koop = data['koop']
    # print(Koop.shape)
    Koop = torch.tensor(np.array(Koop), dtype=torch.float32)

    d = int(data['gramRank'][0][0])
    omegas = data['E']
    if use_log:
        new_omegas = [] 
        for omega in omegas:
            if omega.imag == 0 or np.isclose(omega.imag, np.pi):
                omega = np.exp(omega).real
            else:
                omega = np.exp(omega.real) + omega.imag*1j
            new_omegas.append(omega)
        omegas = np.array(new_omegas)
    # else:
    # print(Koop)

    return Koop, d, omegas

def display_est_embedded(model_encoder, model_decoder, shared_dict, n_traj, return_queue=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from itertools import combinations
    import numpy as np
        
    # print("Displaying trajectories...")
    # start_time = time()

    encoder_dict = shared_dict['encoder_state']
    decoder_dict = shared_dict['decoder_state']
    traj = shared_dict['traj_xt']
    koopman = shared_dict['koopman']
    random = shared_dict['random']

    encoder = model_encoder(traj.shape[2], koopman.shape[0])
    decoder = model_decoder(koopman.shape[0], traj.shape[2])
    encoder.load_state_dict(encoder_dict)
    decoder.load_state_dict(decoder_dict)
    # choose random trajectory(s)
    if random:
        chosen_traj_idx = np.random.randint(0, traj.shape[0], n_traj)
    else:
        chosen_traj_idx = np.arange(0, n_traj)
    # encode the trajectory
    og_traj = traj[chosen_traj_idx, :, :, 0]
    f_pred_traj = torch.zeros(og_traj.shape)
    p_pred_traj = torch.zeros(og_traj.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # convert og into embedding
    og_traj = encoder(og_traj)
    
    # forward through each time step to produce a predicted trajectory
    f_pred_traj[:, 0, :] = og_traj[:, 0, :]
    p_pred_traj[:, 0, :] = og_traj[:, 0, :]
    for traj_n in range(0, n_traj):
        for i in range(0, og_traj.shape[1]-1):
            y_t = og_traj[traj_n, i, :]
            pred_x_t = f_pred_traj[traj_n, i, :]
            # print(encoder(x_t).shape)
            f_pred_traj[traj_n, i+1, :] = koopman @ encoder(pred_x_t)
            p_pred_traj[traj_n, i+1, :] = koopman @ encoder(y_t)

    # convert to numpy
    og_traj = og_traj.detach().numpy()
    f_pred_traj = f_pred_traj.detach().numpy()
    p_pred_traj = p_pred_traj.detach().numpy()


    pairs_dims = list(combinations(range(og_traj.shape[2]), 2))
    n_plots = len(pairs_dims)
    # make it about square
    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Dim {pair[0]} vs Dim {pair[1]}" for pair in pairs_dims])
    
    for idx, (i,j) in enumerate(pairs_dims):
        row = idx // cols + 1
        col = idx % cols + 1
        fig.add_trace(go.Scatter(
            x=og_traj[:, :, i].flatten(),
            y=og_traj[:, :, j].flatten(),
            mode='lines',
            name=f'Original Trajectory',
            line=dict(color='blue', width=4)
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=f_pred_traj[:, :, i].flatten(),
            y=f_pred_traj[:, :, j].flatten(),
            mode='lines',
            name=f'Full Predicted Trajectory',
            line=dict(color='red', width=2)
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=p_pred_traj[:, :, i].flatten(),
            y=p_pred_traj[:, :, j].flatten(),
            mode='lines',
            name=f'Partial Predicted Trajectory (1-Step)',
            line=dict(color='green', width=2)
        ), row=row, col=col)

    fig.update_layout(
        title="Original and Predicted Trajectories in Embedded Space",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        )
    )

    if return_queue is not None:
        return_queue.put([fig, shared_dict['name']])


def display_est_trajectory(model, koopman, traj, random,  n_traj=1, name="Trajectory Estimation", wandb_enabled=False):
    import wandb
    import plotly.graph_objects as go
    
    # print("Displaying trajectories...")
    # start_time = time()
    encoder = model.encoder.cpu()
    decoder = model.decoder.cpu()
    encoder.eval()
    decoder.eval()

    traj = traj

    # choose random trajectory(s)
    if random:
        chosen_traj_idx = np.random.randint(0, traj.shape[0], n_traj)
    else:
        chosen_traj_idx = np.arange(0, n_traj)

    # encode the trajectory
    og_traj = traj[chosen_traj_idx, :, :, 0]
    f_pred_traj = torch.zeros(og_traj.shape)
    p_pred_traj = torch.zeros(og_traj.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # forward through each time step to produce a predicted trajectory
    f_pred_traj[:, 0, :] = og_traj[:, 0, :]
    p_pred_traj[:, 0, :] = og_traj[:, 0, :]
    for traj_n in range(0, n_traj):
        for i in range(0, og_traj.shape[1]-1):
            x_t = og_traj[traj_n, i, :]
            pred_x_t = f_pred_traj[traj_n, i, :]
            print(encoder(pred_x_t).shape)
            f_pred_traj[traj_n, i+1, :] = decoder((koopman @ encoder(pred_x_t)))
            p_pred_traj[traj_n, i+1, :] = decoder((koopman @ encoder(x_t)))

    # convert to numpy
    og_traj = og_traj.detach().numpy()
    f_pred_traj = f_pred_traj.detach().numpy()
    p_pred_traj = p_pred_traj.detach().numpy()

    # Create a Plotly 3D plot
    fig = go.Figure()

    if traj.shape[2] == 2:
        # fix the z axis to 0
        og_traj = np.concatenate((og_traj, np.zeros((n_traj, og_traj.shape[1], 1))), axis=2)
        f_pred_traj = np.concatenate((f_pred_traj, np.zeros((n_traj, f_pred_traj.shape[1], 1))), axis=2)
        p_pred_traj = np.concatenate((p_pred_traj, np.zeros((n_traj, p_pred_traj.shape[1], 1))), axis=2)

    # make everything bigger for better visualization
    # og_traj *= 10
    # f_pred_traj 
    # p_pred_traj *= 10

    for i in range(n_traj):
        # Add the original trajectory
        fig.add_trace(go.Scatter3d(
            x=og_traj[i, :, 0],
            y=og_traj[i, :, 1],
            z=og_traj[i, :, 2],
            mode='lines',
            name=f'{i}th Original Trajectory',
            line=dict(color='blue', width=4)
        ))

        # # Add the predicted trajectory
        # fig.add_trace(go.Scatter3d(
        #     x=f_pred_traj[i, :, 0],
        #     y=f_pred_traj[i, :, 1],
        #     z=f_pred_traj[i, :, 2],
        #     mode='lines',
        #     name=f'{i} Full Predicted Trajectory',
        #     line=dict(color='red', width=2)
        # ))

        # Add the predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=p_pred_traj[i, :, 0],
            y=p_pred_traj[i, :, 1],
            z=p_pred_traj[i, :, 2],
            mode='lines',
            name=f'{i} Partial Predicted Trajectory (1-Step)',
            line=dict(color='green', width=2)
        ))



    # Customize layout
    fig.update_layout(
        title="Original and Predicted Trajectories",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        )
    )

    fig.update_layout(showlegend=False)

    if wandb_enabled:
        wandb.log({"3D {} Trajectories".format(name): fig})

    # save the figure to a file
    fig.write_html("../../data/images/3D {} Trajectories.html".format(name))
    # fig.show()

    # end_time = time()

    # print(f"Time taken to display trajectories: {end_time - start_time} seconds")

def display_simple(data, name="graph"):
    from matplotlib import pyplot as plt

    # print("Displaying trajectories...")
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set(xlabel='x', ylabel='y',
           title=name)
    ax.grid()
    fig.savefig("../../data/images/{}.png".format(name))

    
def display_learned_koop_consistency(predictions, num_iterations):
    '''
    Display the learned Koopman operator from the aux network over epochs by showing mean and variance over batch predictions over time.
    '''
    predictions = predictions.cpu().detach()
    means = predictions.mean(dim=1)  # Shape: (num_iterations, n, n)
    stds = predictions.std(dim=1)  # Shape: (num_iterations, n, n)

    if num_iterations == 1:
        return means, stds
    
    # Plot mean and std of a specific entry in the matrix (e.g., M[0,0])
    plt.figure(figsize=(6,4))
    plt.plot(range(num_iterations), means[:, 0, 0], label="Mean M[0,0]")
    plt.fill_between(range(num_iterations), 
                    means[:, 0, 0] - stds[:, 0, 0], 
                    means[:, 0, 0] + stds[:, 0, 0], alpha=0.3, label="Std Dev")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Convergence of M[0,0] Across Batch")
    plt.savefig("../../data/images/learned_koop_consistency.png")

def plot_lusch(model, params, val_loader, load, name=None):
    wandb_enabled = params['wandb_enabled']

    encoder = model.encoder
    decoder = model.decoder
    auxillary = model.auxillary

    checkpoint = torch.load(load, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    for i, aux_model in enumerate(auxillary):
        aux_model.load_state_dict(checkpoint['auxillary_state_dict'][i])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    batch = next(iter(val_loader))
    # batch = torch.vstack((val_loader.dataset.data[0].unsqueeze(0),val_loader.dataset.data[1].unsqueeze(0)))
    traj= val_loader.dataset.reconstruct_one_batch(batch) 
    # display_est_aux_trajectory(model, params, traj=traj, random=True, n_traj=5, name=name, wandb_enabled=wandb_enabled, use_log=False)
    display_est_aux_trajectory(model, params, val_loader, random=True, n_traj=5, name=name, wandb_enabled=wandb_enabled, use_log=False)

def calculate_mse(dataset, batch, pred_traj):
    og_traj = dataset.reconstruct_one_batch_k(batch, pred_traj.shape[-1])
    raw_mses = []
    for i in range(3,batch.shape[0]):
        raw_MSE = torch.mean(torch.nn.MSELoss(reduction='none')(og_traj[:i], pred_traj[:i]))
        raw_mses.append(raw_MSE)
    
    plt.plot(raw_mses)
    plt.show()


def display_est_aux_trajectory(model, params, test_loader, random,  n_traj=1, name="Trajectory Estimation", wandb_enabled=False, use_log=False):
    import wandb
    import plotly.graph_objects as go

    batch = next(iter(test_loader)) 
    traj = test_loader.dataset.reconstruct_one_batch(batch)

    # print("Displaying trajectories...")
    # start_time = time()
    encoder = model.encoder.cpu()
    decoder = model.decoder.cpu()
    auxillary = model.auxillary

    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']

    # set netwokrs to eval
    encoder.eval()
    decoder.eval()
    for aux in auxillary:
        aux.eval()
        aux.cpu()

    # choose random trajectory(s)
    if random:
        chosen_traj_idx = np.random.randint(0, traj.shape[0], n_traj)
    else:
        chosen_traj_idx = np.arange(0, n_traj)

    # encode the trajectory
    # og_traj = traj[chosen_traj_idx, :, :]
    og_traj = traj.to("cpu")
    # f_pred_traj = full_predict_vec(model, params, traj.to("cpu"), use_log=use_log)
    # f_pred_traj = full_predict_vec(model, params, og_traj, n_steps=params['num_shifts_middle'], use_log=use_log)
    # p_pred_traj = full_predict_vec(model, params, og_traj, n_steps=1, use_log=use_log)
    f_pred_traj_e, f_pred_traj = n_step_predict_vec(model, params, og_traj, n_steps=20, use_log=use_log)
    p_pred_traj_e, p_pred_traj = n_step_predict_vec(model, params, og_traj, n_steps=1, use_log=use_log)
    # test_loader.dataset.reconstruct_one_batch(p_pred_traj, 1,3)
    # f_pred_traj = torch.zeros(og_traj.shape)
    # p_pred_traj = torch.zeros(og_traj.shape)

    fig = go.Figure()
    # # convert to numpy
    og_traj = og_traj.detach().numpy()
    f_pred_traj = f_pred_traj.cpu().detach().numpy()
    p_pred_traj = p_pred_traj.cpu().detach().numpy()

    # print(torch.nn.MSELoss()(torch.tensor(og_traj), torch.tensor(f_pred_traj)))
    # print(torch.nn.MSELoss()(torch.tensor(og_traj), torch.tensor(p_pred_traj)))
    print(torch.nn.MSELoss()(test_loader.dataset.reconstruct_one_batch_k(batch, f_pred_traj.shape[-1])[:, -f_pred_traj.shape[1]:, :], torch.tensor(f_pred_traj)))
    print(torch.nn.MSELoss()(test_loader.dataset.reconstruct_one_batch_k(batch, p_pred_traj.shape[-1])[:, -p_pred_traj.shape[1]:, :], torch.tensor(p_pred_traj)))


    # Create a Plotly 3D plot
    fig = go.Figure()

    if traj.shape[2] == 2:
        # fix the z axis to 0

        og_traj = np.concatenate((og_traj, np.zeros((og_traj.shape[0], og_traj.shape[1], 1))), axis=2)
        f_pred_traj = np.concatenate((f_pred_traj, np.zeros((og_traj.shape[0], f_pred_traj.shape[1], 1))), axis=2)
        p_pred_traj = np.concatenate((p_pred_traj, np.zeros((og_traj.shape[0], p_pred_traj.shape[1], 1))), axis=2)

    # make everything bigger for better visualization
    # og_traj *= 10
    # f_pred_traj 
    # p_pred_traj *= 10

    for i in range(n_traj):
        # Add the original trajectory
        fig.add_trace(go.Scatter3d(
            x=og_traj[i, :, 0],
            y=og_traj[i, :, 1],
            z=og_traj[i, :, 2],
            mode='lines',
            name=f'{i}th Original Trajectory',
            opacity=0.5,
            line=dict(color='blue', width=4)
        ))

        # Add the predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=f_pred_traj[i, :, 0],
            y=f_pred_traj[i, :, 1],
            z=f_pred_traj[i, :, 2],
            mode='lines',
            name=f'{i} Full Predicted Trajectory',
            line=dict(color='red', width=2)
        ))

        # Add the predicted trajectory
        fig.add_trace(go.Scatter3d(
            x=p_pred_traj[i, :, 0],
            y=p_pred_traj[i, :, 1],
            z=p_pred_traj[i, :, 2],
            mode='lines',
            name=f'{i} Partial Predicted Trajectory (5-Step)',
            line=dict(color='green', width=2)
        ))



    # Customize layout
    fig.update_layout(
        title="Original and Predicted Trajectories",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        )
    )

    fig.update_layout(showlegend=False)

    if wandb_enabled:
        wandb.log({"3D {} Trajectories".format(name): fig})

    # save the figure to a file
    fig.write_html("../../data/images/3D {} Trajectories.html".format(name))

def calc_one_step(model, params, states, use_log=False, device=device):
    # print(states.shape)
    # breakpoint()
    encoder = model.encoder.to(device).eval()
    decoder = model.decoder.to(device).eval()
    auxillary = model.auxillary
    for aux in auxillary:
        aux.eval()
        aux.to(device)
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']

    states = states.to(device)
    encoded_x_t = encoder(states)
    if len(encoded_x_t.shape) == 1:
        encoded_x_t = encoded_x_t.unsqueeze(0)

    # print(encoded_x_t.shape)

    nn_omegas_complex = [] 
    nn_omegas_real = []
    for j in range(num_complex_pairs):
        ind = j * 2 
        radius = torch.linalg.norm(encoded_x_t[:, ind:ind+2].unsqueeze(1), dim = 2) ############
        nn_omegas_complex.append(auxillary[j](radius).squeeze(1)) # check whether this should be ind or ind + 2
    for j in range(num_real):
        ind = j + 2*num_complex_pairs
        nn_omegas_real.append(auxillary[j + num_complex_pairs](encoded_x_t[:, ind].unsqueeze(1)).squeeze(1))
            
    if use_log:
        koopman = form_pkoop_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)
    else:
        koopman = form_koopman_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)
    return decoder(torch.bmm(koopman, encoded_x_t.unsqueeze(2)).squeeze(2))


def full_predict(model, params, data, use_log=False):
    start = time()
    data = data[:,:,:,0].to(device) # remove last dimension
    pred_traj = torch.zeros(data.shape).to(device)
    pred_traj[:, 0, :] = data[:, 0, :]
    for n, traj in enumerate(data):
        x_t = traj[0]
        for t in range(len(traj)-1):
            res =  calc_one_step(model, params, x_t, use_log)
            pred_traj[n, t+1] = res
            x_t = res
    end = time()
    print(f"Time taken to predict full trajectory: {end - start}")
    return pred_traj

def full_predict_vec(model, params, data, use_log=False, n_steps=-1, device=device):
    # data = data[:,0].permute(0,2,1).to(device) # remove last dimension
    raise ValueError('full predict vec was not implemented yet with memory')
    batch_size, traj_len, dim = data.shape
    pred_len = traj_len
    if n_steps==-1:
        n_steps = pred_len

    pred_traj = torch.zeros(data.shape).to(device)
    num_sub_traj = int(np.ceil(pred_len / (n_steps+1)))
    last_n_step = pred_len - num_sub_traj*(n_steps+1)
    indices1 = np.arange(0,num_sub_traj*(n_steps+1),n_steps+1)
    # pred_traj[:, 0, :] = data[:, 0, :]
    # pred = data[:, 0, :].reshape(-1, data.shape[2])
    if indices1[-1] + n_steps >= pred_len:
        indices1 = indices1[:-1]
        num_sub_traj = num_sub_traj-1
        last_n_step = pred_len - num_sub_traj*(n_steps+1)
    pred_traj[:,indices1,:] = data[:,indices1,:].to(device)
    for i in range(0,n_steps):
        data_i = pred_traj[:,indices1+i,:].reshape(-1, dim)
        pred = calc_one_step(model, params, data_i, use_log)
        pred_traj[:,indices1+i+1,:] = pred.reshape(-1,num_sub_traj,dim).to(device)

    for j in range(last_n_step):
        data_j= pred_traj[:,-last_n_step+j-1,:].reshape(-1,dim)
        pred_traj[:,-last_n_step+j,:] = calc_one_step(model, params, data_j, use_log)
    return pred_traj


def n_step_predict(model, params, data, n_steps=1, use_log=False):
    data = data[:,:,:,0].to(device) # remove last dimension
    pred_traj = torch.zeros(data.shape).to(device)
    pred_traj[:, 0:n_steps, :] = data[:, 0:n_steps, :]
    for n, traj in enumerate(data):
        for t in range(len(traj)):
            #  predict one step forward
            if t+n_steps >= len(traj)-1:
                break

            x_t = traj[t]
            for i in range(n_steps):
                res =  calc_one_step(model, params, x_t, use_log)
                x_t = res
            # print(res)
            pred_traj[n, t+1] = res
    
    return pred_traj

def n_step_predict_vec(model, params, data, n_steps, use_log=False):
    # data = data[:,:,:,0].permute(0, 2, 1).to(device) # remove last dimension
    
    batch_size, traj_len, dim = data.shape
    memory = params['memory']

    # number_of_preds = traj_len-n_steps if n_steps > memory else traj_len- memory
    number_of_preds = traj_len-n_steps - (memory-1) 
    mem_data = torch.zeros((data.shape[0], number_of_preds,  data.shape[-1] * memory))
    for i in range(number_of_preds):
        inflated_state = data[:, i:i+memory,:].reshape(data.shape[0],-1)
        mem_data[:,i] = inflated_state

    # data = data[:,:,:,0].permute(0, 2, 1).to(device) # remove last dimension
    pred_traj = torch.zeros(data.shape).to(device)
    
    # all_pred = torch.zeros(batch_size,traj_len-n_steps, dim,n_steps)
    all_pred = torch.zeros(batch_size, number_of_preds, dim,n_steps)
    
    pred_traj[:, 0:traj_len-number_of_preds, :] = data[:, 0:traj_len-number_of_preds, :]
    # pred = data[:, :-n_steps, :].reshape(-1, data.shape[2])
    # pred = data[:, :-n_steps, :]# This will be all the "first" states of the memory
    pred = mem_data.reshape(-1,dim*memory)# This will be all the "first" states of the memory
     
    # pred = data[:,:]
    for i in range(n_steps):
        pred = calc_one_step(model, params, pred, use_log)
        # all_pred[:,:,:,i] = pred.reshape(data.shape[0], -1, data.shape[2])
        # data_reshape = torch.zeros((data.shape[0], traj_len-n_steps,  data.shape[-1] * memory))
        # for i in range(traj_len - memory + 1-n_steps):
        #     inflated_state = data[:, i:i+memory,:].reshape(data.shape[0], -1)
        # mem_data[:,i] = inflated_state
        all_pred[:,:,:,i] = pred.reshape(batch_size, -1, dim*memory)[:,:,-dim:]
    # print(pred.shape)
    pred_traj[:, traj_len-number_of_preds:, :] = pred.reshape(batch_size, -1, dim*memory)[:,:,-dim:]
    return pred_traj.to(device), all_pred.to(device)

def display_koop_frobenius_norm(predictions, num_iterations):
    predictions = predictions.cpu().detach()

    means = predictions.mean(dim=1)  # Shape: (num_iterations, n, n)

    frobenius_norms = torch.norm(predictions - means.unsqueeze(1), dim=(2, 3))  # Shape: (num_iterations, batch_size)
    frobenius_mean = frobenius_norms.mean(dim=1)

    # for wandb plotting incrementally instead of all over time
    if num_iterations == 1:
        return frobenius_mean

    plt.figure(figsize=(6,4))
    plt.plot(range(num_iterations), frobenius_mean, label="Avg Frobenius Norm")
    plt.xlabel("Iteration")
    plt.ylabel("Norm")
    plt.title("Convergence of Batch Predictions")
    plt.legend()
    
    plt.savefig("../../data/images/learned_koop_frobenius_norm.png")

def wandb_line_with_error(data, iterations):
    means = data.mean(dim=1)  # Shape: (num_iterations, n, n)
    stds = data.std(dim=1)  # Shape: (num_iterations, n, n)

    # Create the figure once
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the initial mean and std
    line, = ax.plot(iterations, means, label="Mean", color="blue")
    std_fill = ax.fill_between(iterations, means - stds, means + stds, color="blue", alpha=0.2, label="Std (Confidence Interval)")

    # Add labels and legend
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Mean and Standard Deviation Over Time")
    ax.legend()

    # Log the initial plot to wandb
    wandb.log({"mean_std_plot": fig})

    # Update plot over time
    for i in range(1, iterations):
        # Update the mean and std
        line.set_ydata(means[:i])
        std_fill.set_ydata(means[:i] - stds[:i])
        std_fill.set_ydata(means[:i] + stds[:i])
        
        # Log the updated plot
        wandb.log({"mean_std_plot": fig})
