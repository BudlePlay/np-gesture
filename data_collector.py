import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd

df = pd.read_csv('data/train/swing/swing3.csv')

ROW_CNT = 15

fig, ax = plt.subplots(figsize=(15, 8))
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('IMU data collector')

ax.set(xlim=[0, 50], ylim=[0, 50])
ax.set_aspect('auto', adjustable='box')

xdata = [0]
ydata = [0]
line, = ax.plot(xdata, ydata)

x = [i for i in range(len(df))]

ax.set_ylim(-400, 400)

for i in range(1, 4):
    y = df.iloc[:, i] * 20
    ax.plot(x, y, label=str(df.columns[i]), color='r')
    ax.legend()

for i in range(4, 7):
    y = df.iloc[:, i]
    ax.plot(x, y, label=str(df.columns[i]), color='g')
    ax.legend()


def add_point(event):
    if event.inaxes != ax:
        return

    # mouse left click
    if event.button == 1:
        x = event.xdata
        y = event.ydata

        xdata.append(x)
        ydata.append(y)

        line.set_data(xdata, ydata)
        plt.draw()

    # mouse right click
    if event.button == 3:
        xdata.pop()
        ydata.pop()
        line.set_data(xdata, ydata)
        plt.draw()

    # mouse mid click
    if event.button == 2:
        plt.disconnect(cid)
        plt.close()


cid = plt.connect('button_press_event', add_point)
plt.show()
