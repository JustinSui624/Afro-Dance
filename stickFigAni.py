import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

with open("dance_motion.json", "r") as f:
    motion_data = json.load(f)

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)

points, = ax.plot([], [], 'ro')

def update(frame):
    joints = motion_data[frame]["joints"]
    x_vals = [joints[j]["x"] for j in joints]
    y_vals = [joints[j]["y"] for j in joints]
    points.set_data(x_vals, y_vals)
    return points,

ani = animation.FuncAnimation(fig, update, frames=len(motion_data), interval=30)
plt.show()