from utils import Load_cifar10_data, DatasetFolder

# Load CIFAR-10 data
x_train, x_test = Load_cifar10_data()

# Print shapes
print(f"x_train shape: {x_train.shape}")  # Expected: (50000, 3, 32, 32)
print(f"x_test shape: {x_test.shape}")    # Expected: (10000, 3, 32, 32)
