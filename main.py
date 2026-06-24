from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.feature_creation import add_combined_attributes
from src.analysis import print_correlations
from src.visualize import (plot_geographical_data, plot_scatter_matrix)


# load data
housing = load_housing_data()

# split data
strat_train_set, strat_test_set = stratified_split(housing)

# create exploration copy
housing = strat_train_set.copy()

# create new features 
housing = add_combined_attributes(housing)

# print correlations
print_correlations(housing)

# visualize
##plot_geographical_data(housing)
##plot_scatter_matrix(housing)
