import json
from operator import methodcaller


def dt_model_to_dict(dt):
    res = {}
    res["initial_bias"] = dt.initial_bias_
    res["depth"] = dt.depth
    res["n_features"] = dt.n_features_
    res["n_estimators"] = dt.n_estimators
    res["transformer_percentiles"] = list(map(methodcaller("tolist"), dt.transformer.percentiles_))
    res["estimators"] = []
    for estimator in dt.estimators:
        res["estimators"].append({
                "feature": estimator[0],
                "cut": estimator[1],
                "leafs": estimator[2].tolist()
            })
    return res


def dump_dt_pid_to_json(dt_pid, file_name):
    model_as_dict = list(map(dt_model_to_dict, dt_pid.models.values()))
    with open(file_name, 'w') as out_io:
        json.dump(model_as_dict, out_io)
