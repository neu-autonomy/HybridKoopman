import torch

d = 2
encoded_x_tk = torch.arange(0, 100).reshape(10, 2, 5) # Shape: (num_trajectories * num_timesteps, num_components, k)
koop_encoded_x_tn = torch.arange(0, 60).reshape(10, 2, 3) # Shape: (num_trajectories * num_timesteps, num_components, d)

print(encoded_x_tk)
# to compare encoded_x_tk, we need to compare a stacked sliding window of size d with the encoded x_tn
print([encoded_x_tk[:,:, i:i+d] for i in range(0, encoded_x_tk.shape[1]-d+1)])
encoded_x_tk = torch.stack([encoded_x_tk[:, :, i:i+d] for i in range(0, encoded_x_tk.shape[1]-d+1)], dim=1) # Shape: (num_trajectories * num_timesteps, num_components, d)

print(encoded_x_tk.shape)
encoded_x_tk = encoded_x_tk.reshape(*koop_encoded_x_tn.shape) # Shape: (num_trajectories * num_timesteps, num_components, k)
