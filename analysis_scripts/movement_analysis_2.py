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

