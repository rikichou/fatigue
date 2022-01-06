
import binascii
import struct

fpath = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\fatigue_two_stage\mapper_quant\Softmax_43_output0_2_1_1_quant.linear.hex'

with open(fpath, 'r') as fp:
    lines = fp.readlines()
    lines = [a.strip() for a in lines]
    print(lines)
    lines = [int(a, base=16) for a in lines]
    print(lines)
    lines = [a/4096 for a in lines]
    print(lines)
    