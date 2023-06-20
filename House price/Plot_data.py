import plotly
import plotly.graph_objs as go
import pandas as pd

data = pd.read_csv("House price data/train.csv")

trace = go.Scatter(
    x = data['YearBuilt'],
    y = data['SalePrice'],
    mode = 'markers',
    showlegend = True
)
plot_data = [trace]
plotly.offline.plot(plot_data, filename='scatter')