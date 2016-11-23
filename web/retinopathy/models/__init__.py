from svm import SvmLinear2000, SvmLinear5000, SvmPoly2000, SvmPoly5000

_models_by_name = {
   "SvmLinear2000": SvmLinear2000,
   "SvmLinear5000": SvmLinear5000,
   "SvmPoly2000": SvmPoly2000,
   "SvmPoly5000": SvmPoly5000,
}

def get_model_by_name(name):
    return _models_by_name[name]


