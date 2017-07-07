from datetime import datetime
import numpy as np
from pydataset import data as data
from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta
from pyproj import Proj, transform
import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import Figure
from bokeh.models import WMTSTileSource
from bokeh.models import ColumnDataSource

# Problem 1

accidents = pd.read_pickle("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem9/fars_data/final_accidents2.pickle")
accidents.rename(columns={'SP': 'SPEEDING'}, inplace=True)

# Problem 2
from_proj = Proj(init="epsg:4326")
to_proj = Proj(init="epsg:3857")

def convert(longitudes, latitudes):
    """Converts latlon coordinates to meters.
    Inputs:
        longitudes (array-like) : array of longitudes
        latitudes (array-like) : array of latitudes
        Example:
            x,y = convert(accidents.LONGITUD, accidents.LATITUDE)
            """
    x_vals = []
    y_vals = []
    for lon, lat in zip(longitudes, latitudes):
        x, y = transform(from_proj, to_proj, lon, lat)
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

accidents["x"], accidents["y"] = convert(accidents.LONGITUD, accidents.LATITUDE)
            
# Problem 3
drivers = pd.read_pickle("C:/Users/suket/Desktop/Homeworks/Computation/Week2/Problem9/fars_data/final_drivers.pickle")

# Problem 4, Problem 7 (adding Webgl = True)
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import Figure
from bokeh.models import WMTSTileSource


fig = Figure(plot_width=1100, plot_height=650, x_range=(-13000000, -7000000), y_range=(2750000, 6250000),\
             tools=["wheel_zoom", "pan"], active_scroll="wheel_zoom", webgl = True)
fig.axis.visible = False

STAMEN_TONER_BACKGROUND = WMTSTileSource(url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png', attribution=(\
                        'Map tiles by <a href="http://stamen.com">Stamen Design</a>, '\
                        'under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>.'\
                        'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, '\
                        'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>')
)

fig.add_tile(STAMEN_TONER_BACKGROUND)

# Problem 5 (SKip)

# Problem 6
speed = accidents[accidents["SPEEDING"] == 1]
drunk = accidents[accidents["DRUNK_DR"] == 1]
other = accidents[(accidents["DRUNK_DR"] == 0) & (accidents["SPEEDING"] == 0)]

speed_figure = pd.DataFrame({"x_vals" : speed["x"], "y_vals": speed["y"],
               "size":0.3, "fill_color":"red", "fill_alpha":1 })
speed_cir_source = ColumnDataSource(speed_figure)                                 
speed_cir = fig.circle(x="x_vals", y="y_vals", source=speed_cir_source, size="size",
            fill_color="fill_color",fill_alpha="fill_alpha", line_color="red", line_width=3)
   

drunk_figure = pd.DataFrame({"x_vals":drunk["x"], "y_vals":drunk["y"],
               "size":0.3, "fill_color":"yellow", "fill_alpha":1})
drunk_cir_source = ColumnDataSource(drunk_figure)                                 
drunk_cir = fig.circle(x="x_vals", y="y_vals", source=drunk_cir_source, size="size",
            fill_color="fill_color",fill_alpha="fill_alpha", line_color="yellow", line_width=3)
  

other_figure = pd.DataFrame({"x_vals":other["x"], "y_vals":other["y"],
               "size":0.3, "fill_color":"blue", "fill_alpha":1})
other_cir_source = ColumnDataSource(other_figure)                                 
other_cir = fig.circle(x="x_vals", y="y_vals", source=other_cir_source, size="size", 
            fill_color="fill_color",fill_alpha="fill_alpha", line_color="blue", line_width=3)

show(fig)    

# Problem 7 (adding Webgl = True on # Problem 4)
# It is faster than before.

# Problem 8
# Problem 9
# Problem 19
'''
These problems are closely related to problem 5. In fact, after solving problem 5, I can solve these problems. 
'''






































