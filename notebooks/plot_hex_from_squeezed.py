'''
Script to plot squeezed data and corresponding hexagonal data
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys 
import hexagdly
from hexagdly_tools import plot_hextensor

import cnn_datalib as cdl

drop = [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 156, 157, 158, 159, 160, 161, 162, 163, 188, 189, 190, 191, 192, 193, 194, 195, 220, 221, 222, 223, 224, 225, 226, 227, 252, 253, 254, 255, 896, 897, 898, 899, 924, 925, 926, 927, 928, 929, 930, 931, 956, 957, 958, 959, 960, 961, 962, 963, 988, 989, 990, 991, 992, 993, 994, 995, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223]

data_set = cdl.DataSet('/mnt/cnn_data/hessnn_data/h5_data/extendedphase2b5/gamma_20deg_180deg_run84105___phase2b5_desert-ws0_cone5.h5','MC')
event_ids = data_set.unique_index

cuts = {}
for cut, vals in cuts.items():
    event_ids = event_ids.intersection(getattr(data_set, cut)(vals))
event = data_set.get_images(event_ids[0])

cmap='plasma'

plt.ion()

#tel_list = [5]
#cam_type = 'cam2'
#squeezed_data = np.zeros((1, 1, 48, 52))

tel_list = [1, 2, 3, 4]
cam_type = 'cam1'
squeezed_data = np.zeros((1, 4, 32, 36))

for i in event_ids:
    event = data_set.get_images(i)
    
    for n, tel_id in enumerate(tel_list):
        pixels, intensities = event['p_ct'+str(tel_id)], event['i_ct'+str(tel_id)]
        squeezed_data[0, n, :, :] = data_set.ShiftNormed(pixels, intensities, **{'camera_type':cam_type})
    tensor_data = torch.Tensor(squeezed_data)
    
    plot_hextensor(tensor_data, mask=drop)

    saveimg = input('Press "s" to save, "q" to quit, any other key to continue ')
    if saveimg == 's':
        hhf.ExportAllOpenedPlotsAsPDF('event_displays_'+str(event_id)+'.pdf')
    elif saveimg == 'q':
        sys.exit()
