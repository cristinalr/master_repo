import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 26))
train_loss = [0.66999057, 0.36029557, 0.38413976, 0.42127492, 0.63489496, 0.71436626, 0.66562726, 0.3758834, 0.40473573, 
              0.45033826, 0.30142558, 0.60642194, 0.60211666, 0.55068121, 0.51550641, 0.36402602, 0.4651187, 0.64606733, 
              0.37280655, 0.42558669, 0.46332773, 0.50992447, 0.61274348, 0.26933122, 0.55327133]
val_loss = [0.59139939, 0.86346124, 1.01850187, 0.69980617, 0.13665211, 0.50956786, 0.55967, 0.33340962, 1.05882465, 
            0.07380693, 0.95756584, 0.63961536, 0.80938354, 0.25252073, 0.17764909, 0.42367476, 0.97900518, 0.22917516, 
            0.20934265, 0.61145297, 0.17618608, 1.47787513, 0.85912331, 0.51423662, 0.14606852]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Less grid lines
plt.minorticks_off()  # Turn off minor ticks
plt.xticks(epochs)  # Add all epoch points to x-axis for clarity

# Show the plot
plt.show()


import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 26))
train_acc = [
    0.69230769, 0.83333333, 0.83333333, 0.82051282, 0.70512821, 0.67948718,
    0.71794872, 0.76923077, 0.80769231, 0.74358974, 0.84615385, 0.79487179,
    0.71794872, 0.74358974, 0.82051282, 0.79487179, 0.71794872, 0.67948718,
    0.75641026, 0.75641026, 0.79487179, 0.71794872, 0.78205128, 0.84615385,
    0.80769231
]
val_acc = [
    0.7, 0.6, 0.6, 0.7, 0.9, 0.9, 0.8, 0.8, 0.8, 1, 0.8, 0.8, 0.7, 0.9,
    0.9, 0.7, 0.8, 0.8, 0.9, 0.8, 0.8, 0.6, 0.7, 0.9, 1
]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()
