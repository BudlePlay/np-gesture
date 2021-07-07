import pandas as pd
import matplotlib.pyplot as plt

'''
example: 
time, AX, AY, AZ, GX, GY, GZ
12:26:09.013, 0.97, -0.12, -0.10, -15.62, -13.43, -25.27
12:26:09.121, 0.97, -0.17, -0.11, 1.59, -1.65, -5.49
'''

df = pd.read_csv('data/train/0_wait/wait.csv')

ROW_CNT = 15

if len(df) == ROW_CNT:
    print('OK..')
else:
    print('check row cnt')
    print('row cnt : ', len(df))


x = [i for i in range(len(df))]

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
