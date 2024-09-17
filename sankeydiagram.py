import plotly.graph_objects as go

# Define the labels for the nodes
labels = [
    'Total Finance', 'Public Finance', 'Domestic Finance', 'Developmental Finance', 'MEA Finance', 'Private Finance',
    'Corporation', 'Foundation', 'Government', 'Government agency', 'Regional agency', 'Intergovernmental Organisation',
    'International Agency', 'National Institute', 'Non-profit organization', 'Non-profit regional funds',
    'Regional Development Bank', 'Regional Government', 'Research Institute',
    'General Environment Protection', 'Plastics circularity', 'Plastics pollution', 'Post-Secondary Education',
    'Water Supply & Sanitation', 'etc'
]

# Define the source and target node indices and values
sources = [
    0, 0, 0, 0, 0, 0,  # Total Finance to Providers
    1, 1, 1, 1, 1,  # Public Finance to Providers
    2, 2,  # Domestic Finance to Providers
    3, 3,  # Developmental Finance to Providers
    4, 4,  # MEA Finance to Providers
    5, 5   # Private Finance to Providers
]
targets = [
    6, 7, 8, 9, 10,  # Providers
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,  # Sectors/Applications
    22, 23, 24, 25, 26  # Sectors/Applications
]
values = [
    100, 150, 200, 250, 300, 350,  # Total Finance
    80, 120, 180, 220, 270, 320  # Other sources
]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color='black', width=0.5),
        label=labels
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values
    )
)])

# Update layout and show the figure
fig.update_layout(title_text='Finance Flow Sankey Diagram', font_size=10)
fig.show()
