import mlflow

# Create nested runs

# experiment_id = mlflow.create_experiment("experiment7")

with mlflow.start_run(
    run_name="PARENT_RUN",
    experiment_id="6",
    tags={"version": "v1", "priority": "P1"},
    description="parent",
) as parent_run:


    import logging
    import os
    import sys
    import tarfile
    import urllib
    import warnings
    from urllib.parse import urlparse

    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)


    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
            os.makedirs(housing_path, exist_ok=True)
            tgz_path = os.path.join(housing_path, "housing.tgz")
            urllib.request.urlretrieve(housing_url, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=housing_path)
            housing_tgz.close()



    if __name__ == "__main__":
        warnings.filterwarnings("ignore")
        np.random.seed(40)

        # Read the wine-quality csv file from the URL
        DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        HOUSING_PATH = os.path.join("datasets", "housing")
        HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


        fetch_housing_data()

        import pandas as pd

        def load_housing_data(housing_path=HOUSING_PATH):
            csv_path = os.path.join(housing_path, "housing.csv")
            return pd.read_csv(csv_path)
        housing=load_housing_data()

        from zlib import crc32

        from sklearn.model_selection import train_test_split

        with mlflow.start_run(
            run_name="Preparation",
            experiment_id="6",
            description="Data Preparation",
            nested=True,
        ) as child_run:
            train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

            housing["income_cat"] = pd.cut(housing["median_income"],
                                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                           labels=[1, 2, 3, 4, 5])

            from sklearn.model_selection import StratifiedShuffleSplit

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, test_index in split.split(housing, housing["income_cat"]):
                strat_train_set = housing.loc[train_index]
                strat_test_set = housing.loc[test_index]




            for set_ in (strat_train_set, strat_test_set):
                set_.drop("income_cat", axis=1, inplace=True)

            housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
            housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
            housing["population_per_household"]=housing["population"]/housing["households"]

            housing = strat_train_set.drop("median_house_value", axis=1)
            housing_labels = strat_train_set["median_house_value"].copy()

            median = housing["total_bedrooms"].median()
            housing["total_bedrooms"].fillna(median, inplace=True)

            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")

            housing_num = housing.drop("ocean_proximity", axis=1)
            imputer.fit(housing_num)

            X = imputer.transform(housing_num)

            housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                                  index=housing_num.index)

            housing_cat = housing[["ocean_proximity"]]

            from sklearn.preprocessing import OrdinalEncoder
            ordinal_encoder = OrdinalEncoder()
            housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
            housing_cat_encoded[:10]



            from sklearn.preprocessing import OneHotEncoder
            cat_encoder = OneHotEncoder()
            housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
            housing_cat_1hot
            housing_cat_1hot.toarray()

            from sklearn.base import BaseEstimator, TransformerMixin

            rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

            class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
                def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
                    self.add_bedrooms_per_room = add_bedrooms_per_room
                def fit(self, X, y=None):
                    return self  # nothing else to do
                def transform(self, X):
                    rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
                    population_per_household = X[:, population_ix] / X[:, households_ix]
                    if self.add_bedrooms_per_room:
                        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                        return np.c_[X, rooms_per_household, population_per_household,
                                     bedrooms_per_room]

                    else:
                        return np.c_[X, rooms_per_household, population_per_household]

            attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
            housing_extra_attribs = attr_adder.transform(housing.values)

            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            num_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy="median")),
                    ('attribs_adder', CombinedAttributesAdder()),
                    ('std_scaler', StandardScaler()),
                ])

            housing_num_tr = num_pipeline.fit_transform(housing_num)

            from sklearn.compose import ColumnTransformer

            num_attribs = list(housing_num)
            cat_attribs = ["ocean_proximity"]

            full_pipeline = ColumnTransformer([
                    ("num", num_pipeline, num_attribs),
                    ("cat", OneHotEncoder(), cat_attribs),
                ])

            housing_prepared = full_pipeline.fit_transform(housing)

            from sklearn.model_selection import train_test_split

            x_train,x_test,y_train,y_test=train_test_split(housing_prepared,housing_labels)
            alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
            l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5


        # Useful for multiple runs (only doing one run in this sample notebook)
    with mlflow.start_run(
        run_name="Model Creation",
        experiment_id="6",
        description="child",
        nested=True,
    ) as child_run:

        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(x_train, y_train)

        # Evaluate Metrics
        predicted_qualities = lr.predict(x_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

print("parent run:")

print("run_id: {}".format(parent_run.info.run_id))
print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
print("version tag value: {}".format(parent_run.data.tags.get("version")))
print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
print("--")

# Search all child runs with a parent id
query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
results = mlflow.search_runs(experiment_ids=["6"], filter_string=query)
print("child runs:")
print(results[["run_id", "tags.mlflow.runName"]])
