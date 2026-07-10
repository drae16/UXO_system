import sys
if sys.prefix == '/home/drl/poi_dog/.poi':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/drl/poi_dog/Parrot_drone/install/parrot_drone'
