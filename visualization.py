import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_data\wait\wait13.csv')

x = [i for i in range(len(df))]
print(len(df))

for i in range(1, 4):
    y = df.iloc[:, i]
    plt.plot(x, y)
    plt.plot(x, y, label=str(df.columns[i]))
    plt.legend()
    plt.ylim(-5, 5)


plt.title('Accel')
plt.show()

plt.close('all')

for i in range(4, 7):
    y = df.iloc[:, i]
    plt.plot(x, y, label=str(df.columns[i]))
    plt.legend()
    plt.ylim(-400, 400)

plt.title('Gyro')
plt.show()
