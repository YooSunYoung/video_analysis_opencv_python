import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

title_font_dict = {'size': 20}
axis_label_font_dict = {'size': 15}

data = pd.read_csv('result/result.csv', header=0)

data = data.dropna()
diff = data.diff(periods=1)
diff = diff.drop(diff[diff['frame_num'] != 1.0].index)
data = data.drop(diff[diff['frame_num'] != 1.0].index)


diff['left_distance'] = (diff['left_eye_x'].pow(2) + diff['left_eye_y'].pow(2)) ** 0.5
diff['right_distance'] = (diff['right_eye_x'].pow(2) + diff['right_eye_y'].pow(2)) ** 0.5
diff = diff.drop(diff[diff['left_distance'] > 15].index)
diff = diff.drop(diff[diff['right_distance'] > 15].index)
diff['left_distance'].plot.hist(bins=15, alpha=0.8)
diff['right_distance'].plot.hist(bins=15, alpha=0.8)
plt.yscale('log')
plt.text(7, 10000, 'distance less than 15')
plt.text(7, 5000, 'left eye movement mean: {:.3f}'.format(diff['left_distance'].mean()))
plt.text(7, 2500, 'left eye movement mean: {:.3f}'.format(diff['right_distance'].mean()))
plt.xlabel('movement distance', fontdict=axis_label_font_dict)
plt.ylabel('frequency (log scale)', fontdict=axis_label_font_dict)
plt.legend()
plt.savefig('result/eyes_movement_distance.png')


plt.clf()
diff['left_eye_x'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
diff['right_eye_x'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
plt.title('eyes movement on x-axis', fontdict=title_font_dict)
plt.text(0, 3, 'first 200 frames', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
plt.ylabel('eyes movement', fontdict=axis_label_font_dict)
plt.xlabel('frame number', fontdict=axis_label_font_dict)
plt.legend()
plt.savefig('result/eyes_movement_xaxis.png')

plt.clf()
diff['left_eye_y'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
diff['right_eye_y'][:200].plot.line(x=data['frame_num'], figsize=(15, 6))
plt.title('eyes movement on y-axis', fontdict=title_font_dict)
plt.text(20, 3, 'first 200 frames', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})
plt.ylabel('eyes movement', fontdict=axis_label_font_dict)
plt.xlabel('frame number', fontdict=axis_label_font_dict)
plt.legend()
plt.savefig('result/eyes_movement_yaxis.png')

plt.clf()
plt.hist2d(x=diff['left_eye_x'], y=diff['left_eye_y'], bins=20, norm=mpl.colors.LogNorm())
plt.colorbar()
plt.title('left eye movement frequency', fontdict=title_font_dict)
plt.text(5, 6, 'distance less than 15')
plt.text(5, 5, 'movement on x-axis mean: {:.5f}'.format(diff['left_eye_x'].mean()))
plt.text(5, 4, 'movement on y-axis mean: {:.5f}'.format(diff['left_eye_y'].mean()))
plt.xlabel('movement on x-axis', fontdict=axis_label_font_dict)
plt.ylabel('movement on y-axis', fontdict=axis_label_font_dict)
plt.savefig('result/left_eye_movement.png')

plt.clf()
plt.hist2d(x=diff['right_eye_x'], y=diff['right_eye_y'], bins=20, norm=mpl.colors.LogNorm())
plt.colorbar()
plt.title('right eye movement frequency', fontdict=title_font_dict)
plt.text(-15, 15, 'distance less than 15')
plt.text(-15, 14, 'movement on x-axis mean: {:.5f}'.format(diff['right_eye_x'].mean()))
plt.text(-15, 13, 'movement on y-axis mean: {:.5f}'.format(diff['right_eye_y'].mean()))
plt.xlabel('movement on x-axis', fontdict=axis_label_font_dict)
plt.ylabel('movement on y-axis', fontdict=axis_label_font_dict)
plt.savefig('result/right_eye_movement.png')

plt.clf()
plt.hist2d(x=pd.concat([diff['right_eye_x'], diff['left_eye_x']]),
           y=pd.concat([diff['right_eye_y'], diff['left_eye_y']]), bins=20, norm=mpl.colors.LogNorm())
plt.colorbar()
plt.title('eyes movement frequency', fontdict=title_font_dict)
plt.text(-15, 16, 'distance less than 15')
plt.text(-15, 15, 'movement on x-axis mean: {:.5f}'.format((pd.concat([diff['right_eye_x'], diff['left_eye_x']])).mean()))
plt.text(-15, 14, 'movement on y-axis mean: {:.5f}'.format((pd.concat([diff['right_eye_y'], diff['left_eye_y']])).mean()))
plt.xlabel('movement on x-axis', fontdict=axis_label_font_dict)
plt.ylabel('movement on y-axis', fontdict=axis_label_font_dict)
plt.savefig('result/both_eyes_movement.png')
plt.clf()

durations = [[],[],[]]
lengths = [0, 0, 0]
for distances in zip(diff['left_distance'], diff['right_distance'], diff['left_distance']+diff['right_distance']):
    for id, distance in enumerate(distances):
        if distance == 0: lengths[id] += 1
        else:
            if lengths[id] > 0: durations[id].append(lengths[id])
            lengths[id] = 0

left_eye = pd.DataFrame({'left_eye': durations[0]})
right_eye = pd.DataFrame({'right_eye':durations[1]})
both_eyes = pd.DataFrame({'both_eyes':durations[2]})
ax = left_eye.plot.hist(bins=15)
right_eye.plot.hist(bins=15, ax=ax)
both_eyes.plot.hist(bins=15, ax=ax)
plt.title("How long eyes don't move", fontdict=title_font_dict)
plt.yscale('log')
plt.xlabel('duration [frame]', fontdict=axis_label_font_dict)
plt.ylabel('frequency (log scale)', fontdict=axis_label_font_dict)
plt.text(18, 900, 'left eye mean: {:.3f} frames'.format(left_eye['left_eye'].mean()))
plt.text(18, 400, 'right eye mean: {:.3f} frames'.format(right_eye['right_eye'].mean()))
plt.text(18, 200, 'both eyes mean: {:.3f} frames'.format(both_eyes['both_eyes'].mean()))
#plt.text(40, 800, 'movement on y-axis mean: {:.5f}'.format(still.mean()))
plt.legend()
plt.savefig('result/eyes_frozen_duration.png')
plt.clf()

plt.title("How long eyes don't move", fontdict=title_font_dict)
#ax = left_eye[:200].plot.line(figsize=(20, 6))
#right_eye[:200].plot.line(figsize=(20, 6), ax=ax)
both_eyes[:500].plot.line(figsize=(15, 5))
plt.text(0, 8, 'first 500')
plt.xlabel('', fontdict=axis_label_font_dict)
plt.ylabel('duration [frame]', fontdict=axis_label_font_dict)
plt.savefig('result/duration.png')