import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def plot_geographical_data(housing):
    housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"]/100,
    label="population",
    figsize=(10,7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    )

    plt.legend()
    plt.show()

def plot_scatter_matrix(housing):
    attributes = [
        "median_house_value",
        "median_income",
        "total_rooms",
        "housing_median_age"
    ]

    scatter_matrix(
        housing[attributes],
        figsize=(12, 8)
    )

    plt.show()