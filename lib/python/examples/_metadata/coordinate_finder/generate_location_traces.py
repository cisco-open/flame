import os
import random

import yaml

PREV = os.path.dirname(os.path.dirname(__file__))
LOCAL = os.path.dirname(__file__)

in_path = os.path.join(LOCAL, "coordinates.yaml")
out_path = os.path.join(PREV, "location_traces.yaml")
max_time = 600

with open(in_path, "r") as file:
    data = yaml.safe_load(file)
    routes = data["coordinates"]

devices = {}

# Each device randomly selects from an available route.
# Waypoint timings are evenly divided between 0-max_time depending on the
# number of waypoints.
for i in range(300):
    r = random.randrange(0, len(routes))
    name = f"device_{i + 1:03d}"

    devices[name] = []
    steps = max_time / (len(routes[r]) - 1)

    for j, city in enumerate(routes[r]):
        coordinate = {"elapsed_s": j * steps, "lat": city[0], "lon": city[1]}
        devices[name].append(coordinate)

with open(out_path, "w") as file:
    yaml.dump({"traces": devices}, file, default_flow_style=False, sort_keys=False)
