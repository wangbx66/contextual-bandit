import numpy as np
import scipy.sparse as sps
import sys
import os.path

filename = "exploit1.txt"

BASE_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
file1=os.path.join(BASE_DIR, filename)
file_obj = open(file1, encoding='utf-8')
lines = file_obj.readline().strip().lstrip('[').rstrip(']').split(',')
file_write = open("new-"+filename,"w+")
for i in range(len(lines)):
	file_write.write(lines[i].strip());
	file_write.write("\n");




