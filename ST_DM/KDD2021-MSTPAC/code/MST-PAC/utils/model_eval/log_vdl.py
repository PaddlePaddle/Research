import os
import sys
from visualdl import LogWriter

global_step = int(sys.argv[2])
print(global_step)
print('VDL_PATH', os.getenv("VDL_LOG_PATH"))
with open(sys.argv[1]) as f, \
    LogWriter("{}/log_{}".format(os.getenv("VDL_LOG_PATH"), "METRICS"), file_name='vdlrecords.metrics.log') as writer:
    for line in f:
        line = line.strip().split(':')
        writer.add_scalar(tag=line[0], value=float(line[1]), step=global_step)