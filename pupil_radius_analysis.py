import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('result.csv', header=0)

data = data.dropna()
# data = data.map(pd.to_numeric)
diff = data.diff(periods=1)
data = data.drop(diff[diff['frame_num'] != 1.0].index)
diff = diff.drop(diff[diff['frame_num'] != 1.0].index)

data['left_eye_r'].plot.hist(alpha=0.8)
data['right_eye_r'].plot.hist(alpha=0.8)
plt.xlabel('pupil radius')
plt.legend()
plt.savefig('pupil_radius.png')
plt.clf()

data = data.drop(data[data['left_eye_r'] < 32].index)
data = data.drop(data[data['right_eye_r'] < 32].index)

data['left_eye_r'][:150].plot.line(x=data['frame_num'])
data['right_eye_r'][:150].plot.line(x=data['frame_num'], figsize=(15, 6))

plt.xlabel('frame number')
plt.ylabel('pupil radius')

print(data.head())
plt.legend()
plt.savefig('pupil_radius_change.png')