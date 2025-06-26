# import dynatree.solara.select_source as s


def show_load_code(s):
    ans = f"""
# %%
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
import numpy as np
import pandas as pd
sys.path.append("..")
from dynatree.dynatree import DynatreeMeasurement
from dynatree.damping import DynatreeDampedSignal
# %%
m = DynatreeMeasurement(day="{s.day.value}",
                        tree="{s.tree.value}",
                        measurement="{s.measurement.value}",
                        measurement_type="{s.method.value}")
    """
    return ans