from skmodels import (SvmLinear2000, SvmLinear5000, SvmPoly2000, SvmPoly5000,
    RandomForest, LogisticRegression)

_models_by_name = {
   "SvmLinear2000": SvmLinear2000,
   "SvmLinear5000": SvmLinear5000,
   "SvmPoly2000": SvmPoly2000,
   "SvmPoly5000": SvmPoly5000,
   "RandomForest": RandomForest,
   "LogisticRegression": LogisticRegression,
}

def get_model_by_name(name):
    return _models_by_name[name]


