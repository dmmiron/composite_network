import sys
import os
import glob

if __name__ == "__main__":
    composite_yamls = sorted(glob.glob("composite_network_*.yaml"))
    berlin_yamls = sorted(glob.glob("berlin_*.yaml"))
    print composite_yamls, berlin_yamls
    """
    for composite_yaml in composite_yamls:
        os.system("python train.py " + composite_yaml)
    """
    for berlin_yaml in berlin_yamls:
        os.system("python train.py " + berlin_yaml)
