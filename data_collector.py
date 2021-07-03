import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pandas as pd
import sys

df = pd.read_csv('data/AccelGyro3.csv')

ROW_CNT = 15

fig, ax = plt.subplots(figsize=(15, 8))
plt.grid(True)
plt.xlabel('time')
plt.ylabel('y')
plt.title('IMU data collector')

ax.set_aspect('auto', adjustable='box')

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

x_line = []
y_line = []
line, = ax.plot(x_line, y_line)


def remove_line():
    x_line.clear()
    y_line.clear()
    line.set_data(x_line, y_line)
    plt.draw()


def draw_vertical_line(x):
    x_line.append(x)
    y_line.append(500)

    x_line.append(x)
    y_line.append(-500)

    x_line.append(x + 15)
    y_line.append(-500)

    x_line.append(x + 15)
    y_line.append(500)

    line.set_data(x_line, y_line)
    plt.draw()


def add_point(event):
    if event.inaxes != ax:
        return

    # mouse left click
    if event.button == 1:
        x = int(event.xdata)
        y = int(event.ydata)

        remove_line()

        draw_vertical_line(x)

        print(x)
        print(y)

    # mouse right click
    if event.button == 3:
        pass

    # mouse mid click
    if event.button == 2:
        plt.disconnect(cid)
        plt.close()


cid = plt.connect('button_press_event', add_point)

plt.show()
