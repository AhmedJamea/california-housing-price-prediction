from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.preprocessing.transformers import CombinedAttributesAdder
from src.analysis import print_correlations
from src.visualize import (
    plot_geographical_data,
    plot_scatter_matrix
)

housing = load_housing_data()

train_set, test_set = stratified_split(
    housing
)

housing = train_set.copy()

attribute_adder = CombinedAttributesAdder()

housing = attribute_adder.transform(housing)


print_correlations(housing)

plot_geographical_data(housing)

plot_scatter_matrix(housing)