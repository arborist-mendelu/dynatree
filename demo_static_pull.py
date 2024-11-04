from static_pull import DynatreeStaticMeasurement

day = "2022-08-16"
tree = "BK04"
measurement = "M04"
measurement_type = "noc"
data_obj = DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)
print(data_obj.parent.data_pulling)