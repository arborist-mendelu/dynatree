#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:35:00 2024

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""


import unittest
from dynatree.static_pull import DynatreeStaticMeasurement

class Test1(unittest.TestCase):
    def test_vzniku_objektu(self):
        do = DynatreeStaticMeasurement(
            day="2022-04-05",
            tree="BK04",
            measurement="M02",
            optics=False
        )
        self.assertEqual(do.tree,"BK04")

if __name__ == '__main__':
    unittest.main()

# import logging
# lib_dynatree.logger.setLevel(logging.DEBUG)


# [do.plot()] + [i.plot(n) for n,i in enumerate(do.pullings)]
# for i in m.pullings:
#     i.plot()
# a = m.get_static_pulling_data()
# aa = DynatreeStaticPulling(a[0], m.tree)
# # %%
# aa.plot()
# %%
# do.data_optics
# %%
# m.measurement
# a = {}
# a['Measure'] = pull.process_forces(
#     height_of_anchorage=DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
#     height_of_pt=DF_PT_NOTES.at[treeNo,'height_of_pt'],
#     rope_angle=DF_PT_NOTES.at[treeNo,'angle_of_anchorage'],
#     height_of_elastometer=DF_PT_NOTES.at[treeNo,'height_of_elastometer']
#     )
# a['Measure'].columns = [f"{i}_Measure" for i in a['Measure'].columns]
# a['Rope'] = pull.process_forces(
#     height_of_anchorage=DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
#     height_of_pt=DF_PT_NOTES.at[treeNo,'height_of_pt'],
#     height_of_elastometer=DF_PT_NOTES.at[treeNo,'height_of_elastometer']
#     )
# a['Rope'].columns = [f"{i}_Rope" for i in a['Rope'].columns]
# a = pd.concat(list(a.values()))
# ax = a.plot(y=["F_horizontal_Rope", "F_horizontal_Measure"],style = '.')

# # %%
# ax = a.plot(y=["Angle_Measure","Angle_Rope"], style='.')
# ax.grid()
# # m.data
# # # %%
# # a = m.get_static_pulling_data(optics=True)
# # b = m.get_static_pulling_data(optics=False)
# # ax = b[0].plot(y="Force(100)", style='.')
# # a[0].plot(y="Force(100)",ax=ax, style='.')

# # # %%
# # a[0]["Force(100)"] 