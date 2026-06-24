def print_correlations(housing):
    corr_matrix = housing.corr(numeric_only=True)

    print(
        corr_matrix["median_house_value"]
        .sort_values(ascending=False)
    )