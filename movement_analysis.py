import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('result/result.csv', header=0)

data = data.dropna()
diff = data.diff(periods=1)
diff = diff.drop(diff[diff['frame_num'] != 1.0].index)
data = data.drop(diff[diff['frame_num'] != 1.0].index)


diff['left_distance'] = (diff['left_eye_x'].pow(2) + diff['left_eye_y'].pow(2)) ** 0.5
diff['right_distance'] = (diff['right_eye_x'].pow(2) + diff['right_eye_y'].pow(2)) ** 0.5
diff['left_distance'].plot.hist(bins=20, alpha=0.8)
diff['right_distance'].plot.hist(bins=20, alpha=0.8)
plt.xlabel('movement distance')
plt.legend()
plt.savefig('result/eyes_movement_distance.png')

plt.clf()
diff['left_eye_x'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
diff['right_eye_x'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
plt.legend()
plt.savefig('result/eyes_movement_xaxis.png')

plt.clf()
diff['left_eye_y'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
diff['right_eye_y'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
plt.legend()
plt.savefig('result/eyes_movement_yaxis.png')

plt.clf()
plt.hist2d(x=diff['left_eye_x'], y=diff['left_eye_y'])
plt.colorbar()
plt.savefig('result/left_eye_movement.png')

plt.clf()
plt.hist2d(x=diff['right_eye_x'], y=diff['right_eye_y'])
plt.colorbar()
plt.savefig('result/right_eye_movement.png')

plt.clf()
plt.hist2d(x=diff['right_eye_x']+diff['left_eye_x'], y=diff['right_eye_y']+diff['left_eye_y'])
plt.colorbar()
plt.savefig('result/both_eyes_movement.png')