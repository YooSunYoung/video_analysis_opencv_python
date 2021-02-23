import pandas as pd
import matplotlib.pyplot as plt

title_font_dict = {'size': 20}
axis_label_font_dict = {'size': 15}

data = pd.read_csv('result/result.csv', header=0)

data = data.dropna()
# data = data.map(pd.to_numeric)
diff = data.diff(periods=1)
data = data.drop(diff[diff['frame_num'] != 1.0].index)
diff = diff.drop(diff[diff['frame_num'] != 1.0].index)

data['right_eye_r'].plot.hist(alpha=0.8, bins=40)
data['left_eye_r'].plot.hist(alpha=0.8, bins=40)
plt.title('pupil radius', fontdict=title_font_dict)
plt.text(40, 15000, 'left pupil radius mean: {:.3f}'.format(data['right_eye_r'].mean()))
plt.text(40, 14000, 'left pupil radius median: {:.3f}'.format(data['right_eye_r'].median()))
plt.text(40, 13000, 'right pupil radius mean: {:.3f}'.format(data['left_eye_r'].mean()))
plt.text(40, 12000, 'right pupil radius median: {:.3f}'.format(data['left_eye_r'].median()))
plt.xlabel('pupil radius', fontdict=axis_label_font_dict)
plt.ylabel('frequency', fontdict=axis_label_font_dict)
plt.legend()
plt.savefig('result/pupil_radius_histogram.png')
plt.clf()

#data = data.drop(data[data['left_eye_r'] > 32].index)
#data = data.drop(data[data['right_eye_r'] > 32].index)

data['left_eye_r'][:200].plot.line(x=data['frame_num'])
data['right_eye_r'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
plt.title('pupil radius change', fontdict=title_font_dict)
plt.text(0, 20, 'first 200 frames', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
plt.xlabel('frame number', fontdict=axis_label_font_dict)
plt.ylabel('pupil radius', fontdict=axis_label_font_dict)

plt.legend()
plt.savefig('result/pupil_radius_line.png')

plt.clf()

diff['left_eye_r'][:150].plot.line(x=data['frame_num'])
plt.legend()
plt.savefig('result/pupil_radius_change.png')