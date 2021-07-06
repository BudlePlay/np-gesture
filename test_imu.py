import torch

model = torch.load('model.pth', map_location=torch.device('cpu'))
model.eval()

test_input_data = [[-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67], [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67],
                   [-0.08, -0.86, -0.54, 1.89, -0.12, -0.67]]
test_input_data = torch.tensor(test_input_data)
accel = test_input_data[:, 0:3]
gyro = test_input_data[:, 3:7]

accel = accel.reshape(-1)
gyro = gyro.reshape(-1)

out = model(accel.unsqueeze(0), gyro.unsqueeze(0))
print(out)