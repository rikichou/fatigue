import struct
import numpy as np

"""
[0.33694386, 0.16260724, 0.34497467, 0.272598, 0.24577683, 0.16936412, 0.4052017, 0.23417409, 0.420451, 0.22863777, 0.10859343, 0.27901223, 0.48692408, 0.24474595, 0.2131469, 0.17992555, 0.5023406, 0.22164035, 0.1564129, 0.16333498, 0.53938687, 0.28066725, 0.5019938, 0.31877306, 0.38593054, 0.44877234, 0.5116443, 0.5659494, 0.27267328, 0.5803893, 0.3368946, 0.33180216, 0.35108715, 0.5286323, 0.26155692, 0.18843783, 0.22538936, 0.20809042, 0.2679227, 0.3011071, 0.18253815, 0.17735703, 0.5401628, 0.105130956, 0.117503226, 0.35454166, 0.14025916, 0.16518816, 0.15395905, 0.35690457, 0.3044631, 0.47841704, 0.24743773, 0.3560374, 0.23847581, 0.34936395, 0.39382717, 0.42635095, 0.44736832, 0.2705193, 0.33320194, 0.36136422, 0.33003587, 0.14665243, 0.21185641, 0.19997735, 0.5276178, 0.267062, 0.30228305, 0.77496535, 0.3068626, 0.5191417, 0.24040042, 0.3038338, 0.62930065, 0.6786277, 0.3902934, 0.5592512, 0.39748767, 0.47518986, 0.24882819, 0.39541575, 0.7107541, 0.1534659, 0.36376658, 0.50157106, 0.41083416, 0.46955657, 0.43735132, 0.14853708, 0.33729655, 0.2897546, 0.39710727, 0.12642647, 0.12101772, 0.4704896, 0.20588402, 0.26595116, 0.21658707, 0.29128718, 0.23557422, 0.66206944, 0.50319326, 0.3796948, 0.32402647, 0.62216187, 0.46964404, 0.3350887, 0.29720503, 0.41971007, 0.15107864, 0.5392459, 0.38101348, 0.31951952, 0.2577134, 0.31694493, 0.60630184, 0.4513625, 0.1650606, 0.29833624, 0.20846568, 0.3954671, 0.24474747, 0.25035322, 0.18979414, 0.39343262, 0.17719008, 0.26442406]
47
0.9422097
"""

bin_path = '/home/ruiming/workspace/pro/fatigue/data/test/bins/CNN_43_OUT0.bin'

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

float_features_lists = []
int16_features_lists = []

with open(bin_path, 'rb') as fp:

    for i in range(128):
        info1 = fp.read(2)
        
        info1 = struct.unpack('<h', info1)[0]
        
        int16_features_lists.append(info1)
        float_features_lists.append(info1 / (2**15))

caffe_output = [-0.068538815, 0.030214332, -0.09281872, -0.09276931, -0.025088673, 0.05985368, 0.031862307, -0.12744951, -0.042850498, 0.2891812, 0.061912872, 0.0065802485, 0.067591816, -0.0019157437, -0.036146346, -0.036415756, 0.122296676, 0.055363785, -0.14266893, 0.10266328, 0.12811954, -0.041409213, 0.034873977, 0.19426462, 0.07464902, 0.02436021, -0.15745926, 0.0223604, 0.12016702, -0.32098594, -0.23028168, -0.021731343, -0.19648124, 0.1061582, 0.02383303, 0.048134226, -0.15408073, -0.028216239, 0.08362739, 0.22947827, 0.09089055, -0.0057037547, 0.1574323, 0.04231924, -0.24319667, 0.12842967, -0.17426436, 0.017402079, 0.10726939, -0.24007285, 0.07424011, -0.18052348, -0.0780444, 0.14884232, 0.06905798, 0.009250738, -0.036246166, 0.0023647249, -0.018075932, 0.20258163, -0.0541235, 0.1384747, 0.28140742, -0.18167953, -0.026445523, 0.12453222, 0.037108734, 0.059595317, -0.14076485, -0.15754968, -0.049869075, 0.12314624, -0.1599174, -0.16146192, 0.15709928, 0.106035724, 0.09401655, 0.14829905, 0.06288643, -0.104058355, -0.117733456, -0.15206435, -0.048513472, -0.09780621, -0.19679075, -0.10057699, -0.16171026, -0.0074851513, -0.10985569, -0.055256814, 0.11466329, -0.059535265, 0.019270796, -0.0039372817, 0.23852488, -0.1328957, -0.034103543, 0.14929572, 0.20149854, -0.02496742, -0.29200894, 0.12755182, -0.34550107, 0.10755433, -0.022196246, -0.09007372, 0.027993273, 0.1693854, -0.24270439, -0.06339635, 0.068045825, 0.05628929, -0.11935896, 0.0073122717, 0.11943443, 0.25675657, 0.21649355, -0.07111011, -0.044282027, -0.21198702, -0.017069405, 0.09741692, 0.107574984, 0.19438815, 0.23286557, -0.09636815, 0.07505941, 0.07786899]
caffe_output = np.array(caffe_output)

#int16_features_lists = np.array([-0.066315,0.026367,-0.088959,-0.086975,-0.028839,0.058197,0.037537,-0.119781,-0.042725,0.286469,0.060364,0.006073,0.069519,-0.005005,-0.036255,-0.033356,0.117371,0.053650,-0.140900,0.104309,0.132111,-0.039642,0.040161,0.193726,0.077332,0.026123,-0.164337,0.026703,0.121368,-0.325073,-0.232880,-0.023224,-0.188538,0.108459,0.019562,0.043671,-0.156677,-0.028412,0.079895,0.227356,0.085876,-0.000244,0.154114,0.043549,-0.244690,0.129395,-0.171844,0.011566,0.109131,-0.242920,0.078522,-0.181824,-0.077026,0.153381,0.069733,0.010925,-0.029846,-0.011963,-0.009552,0.199738,-0.052399,0.148895,0.281158,-0.183929,-0.025482,0.128845,0.034241,0.063202,-0.142517,-0.165314,-0.054718,0.117615,-0.159790,-0.166656,0.155487,0.105469,0.095306,0.144928,0.071045,-0.099365,-0.119202,-0.149689,-0.044281,-0.099182,-0.195435,-0.103363,-0.169281,-0.000946,-0.118866,-0.060486,0.116302,-0.060181,0.018982,-0.002930,0.237671,-0.134430,-0.032074,0.155945,0.200226,-0.026520,-0.288086,0.122253,-0.351654,0.107056,-0.032166,-0.089661,0.026978,0.167480,-0.241730,-0.066803,0.063446,0.053925,-0.129333,0.014648,0.122589,0.255707,0.218597,-0.069427,-0.051758,-0.220734,-0.011688,0.101166,0.109833,0.198517,0.236389,-0.090790,0.080414,0.070007])
int16_features_lists = np.array(int16_features_lists)

print(cosine_similarity(caffe_output, int16_features_lists))


