from SznaierKoopman import *
import itertools
import numpy as  np

def experiments(mode="train"):
    exp_names = ["FluidFlowBox"]
    exp_vers = ["train1_x", "test_x"]
    num_epochs = 500
    batch_size = 128
    learning_rate = 1e-3
    traj_len = [400, 400, 400, 400]
    k = 40
    aux = False
    lambda1 = [0.1]
    lambda2 = [0.4]
    lambda3 = [0.9]
    # 0.1 0.9 0.4
    multiprocessing.set_start_method('spawn')


    if mode == "train":
        hyperparam_sets = list(itertools.product([num_epochs], [batch_size], [learning_rate], [k], lambda1, lambda2, lambda3))
        for exp_name in exp_names:
            koopman_path = "../data/koopmans/{}_koopman.mat".format(exp_name)
            for hyperparams in hyperparam_sets:
                exp_name_index = exp_names.index(exp_name)
                num_epochs, batch_size, learning_rate, k, l1, l2, l3 = hyperparams
                print("Running Experiment {} with hyperparameters: num_epochs: {}, batch_size: {}, learning_rate: {}, traj_len: {}, k: {}, l1: {}, l2: {}, l3: {}".format(exp_name + "_" + exp_vers[0], num_epochs, batch_size, learning_rate, traj_len, k, l1, l2, l3))
                run(exp_name, exp_vers, num_epochs, batch_size, learning_rate, traj_len[exp_name_index], k, koopman_path, l1, l2, l3, aux)

    # elif mode == "test":
    #     for exp_name in exp_names:
    #         test_models(exp_name + "_" + exp_vers[1])



def embedding_2d_plot(model_path):
    encoder = Encoder()
    encoder.load_state_dict(torch.load("./models/" + model_path + "_encoder.pth"))


 
if __name__ == "__main__":
    # load from path
    exp_name = "FluidFlowBox"
    exp_ver = ["train1_x", "val_x", "test_x"]
    koopman_path = "../data/koopmans/{}_koopman.mat".format("FluidFlowBox")
    model_path = "FluidFlowBox_['train1_x', 'test_x']_azure-durian-1054"
    aux = True
    traj_len = 400
    k = 40
    batch_size = 100
  
    koopman = load_koopman(koopman_path)

    train_dataset = pd.read_csv('../data/csv/{}_{}.csv'.format(exp_name, exp_ver[0]), delimiter=',', header=None, dtype=np.float64).values
    train_dataset = ds.KoopData(train_dataset, traj_len=traj_len, k=k)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = pd.read_csv('../data/csv/{}_{}.csv'.format(exp_name, exp_ver[1]), delimiter=',', header=None, dtype=np.float64).values
    val_dataset = ds.KoopData(val_dataset, traj_len=traj_len, k=k)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = pd.read_csv('../data/csv/{}_{}.csv'.format(exp_name, exp_ver[2]), delimiter=',', header=None, dtype=np.float64).values
    test_dataset = ds.KoopData(test_dataset, traj_len=traj_len, k=k)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    encoder = Encoder(train_dataset[0].shape[1], koopman.shape[0]).to(device)
    decoder = Decoder(koopman.shape[0], train_dataset[0].shape[1]).to(device)
    auxillary = Auxillary(*koopman.shape).to(device)
    encoder.load_state_dict(torch.load("./models/" + model_path + "_encoder.pth"))
    decoder.load_state_dict(torch.load("./models/" + model_path + "_decoder.pth"))
    auxillary.load_state_dict(torch.load("./models/" + model_path + "_auxillary.pth"))
    koopman = load_koopman(koopman_path).to(device)
    
    encoder.eval()
    decoder.eval()
    auxillary.eval()

    n_traj = 100

    acc = []
    for j, loader in enumerate([train_loader, val_loader, test_loader]):
        for i, data in enumerate(loader):
            if aux:
                og_traj = data[:,:,:,0].to(device)
                og_data = og_traj.to(device)

                print(og_data.shape)
                encoded = encoder(og_data)
                thru_aux = auxillary(encoded)
                decoded = decoder(thru_aux)
                
                if j >= len(acc):
                    acc.append(torch.nn.MSELoss(reduction="sum")(decoded, og_traj).item())
                acc[j] += torch.nn.MSELoss(reduction="sum")(decoded, og_traj).item()
            else:
                og_data = data.to(device)
                og_traj = og_data[:,:,:,0]
                koopman.to(device)

                p_pred_traj = torch.zeros(n_traj, og_traj.shape[1], og_traj.shape[2]).to(device)
                p_pred_traj[:, 0, :] = og_traj[:, 0, :]
                for traj_n in range(0, n_traj):
                    for i in range(0, og_traj.shape[1]-1):
                        x_t = og_traj[traj_n, i, :]
                        p_pred_traj[traj_n, i+1, :] = decoder((koopman @ encoder(x_t)))

                # calculate the accuracy
                if j >= len(acc):
                    acc.append(torch.nn.MSELoss(reduction='sum')(p_pred_traj, og_traj).item())
                acc[j] += torch.nn.MSELoss(reduction='sum')(p_pred_traj, og_traj).item()

                print("Loader: {}, Iteration: {}, Loss: {}".format(j, i, acc[j]/(i+1)))

    print("Train Loss: {}, Val Loss: {}, Test Loss: {}".format(acc[0], acc[1], acc[2]))

            
    experiments("train")
