import matplotlib.pyplot as plt
import numpy as np
import time

# Start with an empty array
data = np.array([])

plt.ion()  # Turn on interactive mode
fig = plt.figure()  # Create a new figure
ax = fig.add_subplot(111)  # Add a subplot
line1, = ax.plot(data)  # Plot the initial data

# Update the plot with new data every 0.2 seconds
for i in range(100):
    new_data = np.random.rand()  # Generate a new data point
    data = np.append(data[-19:], new_data)  # Update the data array, keep only the last 20 points
    line1.set_ydata(data)  # Update the plot
    line1.set_xdata(range(len(data)))  # Update the x-axis
    ax.relim()  # Recalculate limits
    ax.autoscale_view(True, True, True)  # Rescale the plot
    plt.draw()  # Redraw the plot
    plt.pause(0.2)  # Pause for 0.2 seconds

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
