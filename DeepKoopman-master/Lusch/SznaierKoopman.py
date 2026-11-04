from scipy.io import loadmat, savemat
import scipy.io
import numpy as np
import pandas as pd
import torch
import dataset as ds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import multiprocessing
import matplotlib.pyplot as plt
from time import time
import cProfile
import pstats


def load_koopman(path=None):
    if path is None:
        path = '../data/koopmans/DiscreteSpectrumExample_observables.mat'
    data = loadmat(path)
    Koop = data['pK'] if 'pK' in data else data['koop']  # older exports stored pK, the SDP results store koop
    # print(Koop.shape)
    Koop = torch.tensor(np.array(Koop), dtype=torch.float32)
    # print(Koop)
    

    return Koop

def display_est_embedded(shared_dict, n_traj, return_queue=None):
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

    encoder = Encoder(traj.shape[2], koopman.shape[0])
    decoder = Decoder(koopman.shape[0], traj.shape[2])
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


def display_est_trajectory(shared_dict, n_traj=1, return_queue=None):
    import plotly.graph_objects as go
    
    # print("Displaying trajectories...")
    # start_time = time()

    encoder_dict = shared_dict['encoder_state']
    decoder_dict = shared_dict['decoder_state']
    traj = shared_dict['traj_xt']
    koopman = shared_dict['koopman']
    random = shared_dict['random']

    encoder = Encoder(traj.shape[2], koopman.shape[0])
    decoder = Decoder(koopman.shape[0], traj.shape[2])
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
    
    # forward through each time step to produce a predicted trajectory
    f_pred_traj[:, 0, :] = og_traj[:, 0, :]
    p_pred_traj[:, 0, :] = og_traj[:, 0, :]
    for traj_n in range(0, n_traj):
        for i in range(0, og_traj.shape[1]-1):
            x_t = og_traj[traj_n, i, :]
            pred_x_t = f_pred_traj[traj_n, i, :]
            # print(encoder(x_t).shape)
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

    if return_queue is not None:
        return_queue.put([fig, shared_dict['name']])

    # end_time = time()

    # print(f"Time taken to display trajectories: {end_time - start_time} seconds")

# Define the Autoencoder class
class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, out_dim)
        )

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        return encoded
    
class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)        
            )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded


class Auxillary(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Auxillary, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        aux = self.linear(x)
        return aux

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lusch_loss(epoch, encoder, decoder, x_t, k, koop, lambdas):
    start_time = time()
    forward_loss = 0
    linearity_loss = 0
    reconstruction_loss = 0
    for i in range(1, k+1):
        for j in range(0, x_t.shape[1]):
            x_tn = x_t[:, j, :, 0]
            x_tk = x_t[:, j, :, i]

            # if i == 1 and j == 0:
            #     # print((torch.linalg.matrix_power(koop, i) @ encoder(x_tn).T).T.shape)
            #     print(((torch.linalg.matrix_power(koop, i) @ encoder(x_tn).T).T)[0])

            # Reconstruction loss
            reconstruction_loss_i = nn.MSELoss()(decoder(encoder(x_tn)), x_tn)
            # Linearity loss
            linearity_loss_i = nn.MSELoss()(encoder(x_tk), (torch.linalg.matrix_power(koop, i) @ encoder(x_tn).T).T)
            # Koopman forward loss
            forward_loss_i =  nn.MSELoss()(x_tk, decoder((torch.linalg.matrix_power(koop, i) @ encoder(x_tn).T).T))

        reconstruction_loss += lambdas[0] * reconstruction_loss_i
        linearity_loss += lambdas[1]**(i+1) * linearity_loss_i
        forward_loss += lambdas[2]**(i+1) * forward_loss_i


    full_loss = reconstruction_loss + linearity_loss + forward_loss
    # print("Reconstruction, Forward, Linearity Loss: ", reconstruction_loss, forward_loss, linearity_loss)
    wandb.log({"Epoch":epoch+1, "Reconstruction Loss": reconstruction_loss, "Forward Loss": forward_loss, "Linearity Loss": linearity_loss, "Total Loss": full_loss})
    
    end_time = time()
    # print(f"Time taken to compute loss: {end_time - start_time} seconds")

    return full_loss

def vec_lusch_loss(epoch, encoder, decoder, x_t, k, koop, lambdas, aux=None):
    start_time = time()
    batched_x_t = x_t.reshape(-1, x_t.shape[2], x_t.shape[3])  # Shape: (num_trajectories * num_timesteps, num_components, k)

    # print("batched_x_t shape: ", batched_x_t.shape)
    # extract reference states
    x_tn = batched_x_t[:, :, 0] # Shape: (num_trajectories * num_timesteps, num_components, first_state)
    x_tk = batched_x_t[:, :, 1:].permute(2,0,1) # Shape: (num_trajectories * num_timesteps, num_components, kth state)

    # print("x_tn shape: ", x_tn.shape)
    # print("x_tk shape: ", x_tk.shape)

    encoded_x_tn = encoder(x_tn) # Shape: (num_trajectories * num_timesteps, num_components, out_dim)
    # the encoder cant take all ks, so flatten the k dimension
    # print("encoded_x_tn shape: ", encoded_x_tn.shape)

    

    encoded_x_tk = encoder(x_tk.reshape(-1, x_tk.shape[-1]))
    # print("encoded_x_tk shape: ", encoded_x_tk.shape)
    if aux is None:
        koop_stacked = torch.stack([torch.linalg.matrix_power(koop, k) for k in range(0, k)], dim=0).to(device) # Shape: (k, num_components, num_components)
        # print("koop_stacked shape: ", koop_stacked.shape)

        koop_encoded_x_tn = (koop_stacked @ encoded_x_tn.permute(1,0)).permute(0,2,1) # Shape: (num_trajectories * num_timesteps, num_components, k)
    
    else:
        # # if epoch is before 100, freeze the auxillary network
        # if epoch < 100:
        #     aux.requires_grad_(False)
        # else:
        #     aux.requires_grad_(True)

        koop_encoded_x_tn = torch.zeros((k, encoded_x_tn.shape[0], encoded_x_tn.shape[1])).to(device)
        koop_encoded_x_tn_i = encoded_x_tn

        for i in range(0, k):
            # send through auxillary network that many times
            koop_encoded_x_tn_i = aux(koop_encoded_x_tn_i)
            
            # print(koop_encoded_x_tn.shape)
            # print(koop_encoded_x_tn_i.shape)
            koop_encoded_x_tn[i, :, :] = koop_encoded_x_tn_i.unsqueeze(0)

    # apply the decoder to the non-koopman encoded states
    decoded_encoded_x_tn = decoder(encoded_x_tn) # Shape: (num_trajectories * num_timesteps, num_components, num_components)

    # apply the decoder to the koopman encoded states - cant do all ks at once, so flatten the k dimension
    decoded_koop_encoded_x_tk = decoder(koop_encoded_x_tn.reshape(-1, koop_encoded_x_tn.shape[-1])) # Shape: (num_trajectories * num_timesteps, num_components, num_components)

    # print(decoded_encoded_x_tn.shape, x_tn.shape)
    # print(encoded_x_tk.shape, koop_encoded_x_tn.shape)
    # print(decoded_koop_encoded_x_tk.shape, x_tk.shape)

    reconstruction_loss = nn.MSELoss(reduction="none")(decoded_encoded_x_tn, x_tn)

    linearity_loss = nn.MSELoss(reduction='none')(encoded_x_tk.reshape(*koop_encoded_x_tn.shape), koop_encoded_x_tn)

    forward_loss = nn.MSELoss(reduction='none')(decoded_koop_encoded_x_tk.reshape(*x_tk.shape), x_tk)

    lambda2 = torch.pow(lambdas[1], torch.arange(1, k+1)).to(device)
    lambda3 = torch.pow(lambdas[2], torch.arange(1, k+1)).to(device)

    # mean across all dims but given dim
    reconstruction_loss = torch.mean(reconstruction_loss)
    linearity_loss = torch.mean(linearity_loss, dim = [1,2])
    forward_loss = torch.mean(forward_loss, dim = [1,2])

    reconstruction_loss = torch.sum(lambdas[0] * reconstruction_loss)
    # dot product of the linearity loss and the lambda2 vector
    linearity_loss = torch.sum(torch.dot(linearity_loss, lambda2))
    forward_loss = torch.sum(torch.dot(forward_loss, lambda3))

    full_loss = reconstruction_loss + linearity_loss + forward_loss
    # print("Reconstruction, Forward, Linearity Loss: ", reconstruction_loss, forward_loss, linearity_loss)

    wandb.log({"Epoch":epoch+1, "Reconstruction Loss": reconstruction_loss, "Forward Loss": forward_loss, "Linearity Loss": linearity_loss, "Total Loss": full_loss})

    end_time = time()
    # print(f"Time taken to compute loss: {end_time - start_time} seconds")

    return full_loss


def run(exp_name, exp_ver, num_epochs, batch_size, learning_rate, traj_len, k, koopman_path, l1, l2, l3, aux=NotImplemented):
    run_start_time = time()
    wandb.init(project="KoopmanGrowth", 
            config={"learning_rate": learning_rate, 
                "batch_size": batch_size, 
                "traj_len": traj_len, 
                "k": k,
                "l1": l1,
                "l2": l2,
                "l3": l3,
                "aux_setting": "No Auxillary Network"})

    wandb_name = wandb.run.name
    wandb.run.name = exp_name + "-" + wandb.run.name

    koopman = load_koopman(koopman_path)
    koopman = koopman.to(device)

    l = [l1, l2, l3]

    # Load the dataset csv
    train_dataset = pd.read_csv('../data/csv/{}_{}.csv'.format(exp_name, exp_ver[0]), delimiter=',', header=None, dtype=np.float64).values
    train_dataset = ds.KoopData(train_dataset, traj_len=traj_len, k=k)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = pd.read_csv('../data/csv/{}_{}.csv'.format(exp_name, exp_ver[1]), delimiter=',', header=None, dtype=np.float64).values
    val_dataset = ds.KoopData(val_dataset, traj_len=traj_len, k=k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # print(train_dataset[0].shape[1])
    # Initialize the autoencoder, loss function, and optimizer
    encoder = Encoder(train_dataset[0].shape[1], koopman.shape[0]).to(device)
    decoder = Decoder(koopman.shape[0], train_dataset[0].shape[1]).to(device)

    if aux is not None:
        auxillary = Auxillary(*koopman.shape).to(device)
        # with torch.no_grad():
        #     auxillary.linear.weight.copy_(koopman)

        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(auxillary.parameters()), lr=learning_rate)
    else:
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    # Training loop
    prev_epoch_loss = 0
    for epoch in range(num_epochs):
        epoch_start_time = time()
        total_loss = 0
        for traj_xt in train_loader:
            traj_xt = traj_xt.to(device)

            if aux is not None:
                send_aux = auxillary
                # if epoch < 100:
                #     list(auxillary.parameters())
            else:
                send_aux = None

            loss = vec_lusch_loss(epoch, encoder, decoder, traj_xt, k, koopman, l, send_aux)
            # loss2 = lusch_loss(epoch, encoder, decoder, traj_xt, k, koopman, l)


            # # TEST
            # rand_3x3 = torch.tensor([2.0, -2.0, 2.0]).to(device)
            # koop_test = koopman @ rand_3x3
            
            # with torch.no_grad():
            #     aux_test = auxillary(rand_3x3)

            #     print(koop_test,"\n", aux_test)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


        # if (epoch+1) % 5 == 0:
        #     next_val = next(iter(val_loader))
        #     next_val = next_val.to(device)
        #     val_loss = vec_lusch_loss(epoch, encoder, decoder, next_val, k, koopman, l)
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():f}, Val Loss: {val_loss.item():f}')

        #     wandb.log({"epoch": epoch, "Validation Loss": val_loss})


        if (epoch + 1) % 100 == 0:
            manager = multiprocessing.Manager()
            return_dict = manager.Queue()
            shared_dict = manager.dict()
            # if there are no mroe than 3 processes running, start a new one
            if len(multiprocessing.active_children()) < 3:
                shared_dict['encoder_state'] = {k: v.cpu() for k, v in encoder.state_dict().items()}
                shared_dict['decoder_state'] = {k: v.cpu() for k, v in decoder.state_dict().items()}
                shared_dict['name'] = "Train"
                shared_dict['traj_xt'] = traj_xt.cpu()
                shared_dict['koopman'] = koopman.cpu()
                shared_dict['random'] = False

                # process = multiprocessing.Process(
                #     target=display_est_trajectory, 
                #     args=(shared_dict, 8, return_dict))
                test_data = next(iter(val_loader)).cpu()
                shared_dict['traj_xt'] = test_data
                shared_dict['name'] = "Test"
                process2 = multiprocessing.Process(
                    target=display_est_trajectory, 
                    args=(shared_dict, 6, return_dict))
                
                # process.start()
                process2.start()
                # process.join()
                process2.join()

                if not return_dict.empty():
                    fig, name = return_dict.get()
                    # print("occurs")
                    wandb.log({"3D {} Trajectories".format(name): fig})


        if abs(total_loss - prev_epoch_loss) < 1e-3:
            print("Converged at epoch: ", epoch)
            converged_time = time()
            print(f"Time taken to converge: {converged_time - run_start_time} seconds")
            break


        prev_epoch_loss = total_loss

        # Print loss for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/ len(train_loader):f}')
        epoch_end_time = time()
        # print(f"Time taken for epoch: {epoch_end_time - epoch_start_time} seconds")


    
    wandb.finish()

    # Save the trained model
    torch.save(encoder.state_dict(), f'./models/{exp_name}_{exp_ver}_{wandb_name}_encoder.pth')
    torch.save(decoder.state_dict(), f'./models/{exp_name}_{exp_ver}_{wandb_name}_decoder.pth')
    if aux is not None:
        torch.save(auxillary.state_dict(), f'./models/{exp_name}_{exp_ver}_{wandb_name}_auxillary.pth')
    
    run_end_time = time()
    print(f"Total time taken for run: {run_end_time - run_start_time} seconds")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    # # run cprofile
    # cProfile.run('run("DiscreteSpectrumExample", ["train1_x", "val_x"], "test", 200, 64, 1e-3, 400, 7, "../data/koopmans/DiscreteSpectrumExample_observables.mat", 0.1, 0.3, 0.6)')

    wandb.init(project="KoopmanGrowth", config={"learning_rate": 1e-3, 
                                                                     "batch_size": 64, 
                                                                     "traj_len": 400, 
                                                                     "k": 7})
    

    # MSE_loss = nn.MSELoss()
    
    exp_name = "DiscreteSpectrumExample"
    wandb_name = wandb.run.name
    wandb.run.name = exp_name + "-" + wandb.run.name
    exp_ver = ["train1_x", "val_x"]
    num_epochs = 200
    batch_size = 64
    learning_rate = 1e-3
    traj_len = 400
    k = 7
    koopman_path = "../data/koopmans/DiscreteSpectrumExample_observables.mat"
    l1 = 0.1
    l2 = 0.3
    l3 = 0.6

    run(exp_name, exp_ver, wandb_name, num_epochs, batch_size, learning_rate, traj_len, k, koopman_path, l1, l2, l3)

    # cProfile.run('run("DiscreteSpectrumExample", ["train1_x", "val_x"], "test", 200, 64, 1e-3, 400, 7, "../data/koopmans/DiscreteSpectrumExample_observables.mat", 0.1, 0.3, 0.6)', 'profile.prof')

    # p = pstats.Stats('profile.prof')
    # p.strip_dirs().sort_stats('cumulative').print_stats()

