from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from .transformers import CombinedAttributesAdder


def build_numerical_pipeline():

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("scaler", StandardScaler())
    ])

def build_full_pipeline(housing):
    
    numerical_attributes = housing.drop(
        "ocean_proximity",
        axis=1
    ).columns.tolist()

    categorical_attributes = [
        "ocean_proximity"
    ]

    return ColumnTransformer([
        (
            "num",
            build_numerical_pipeline(),
            numerical_attributes
        ),
        (
            "cat",
            OneHotEncoder(),
            categorical_attributes
        )
    ])