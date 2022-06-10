# Flame SDK

## Selector
Users are able to implement new selectors in `lib/python/flame/selector/` which should return a dictionary with keys corresponding to the active trainer IDs (i.e., agent IDs). After implementation, the new selector needs to be registered into both `lib/python/flame/selectors.py` and `lib/python/flame/config.py`.
### Currently Implemented Selectors
1. Naive (i.e., select all)
```json
"selector": {
    "sort": "default",
    "kwargs": {}
}
```
2. Random (i.e, select k out of n local trainers)
```json
"selector": {
    "sort": "random",
    "kwargs": {
        "k": 1
    }
}
```

## Optimizer (i.e., aggregator of FL)
Users can implement new server optimizer, when the client optimizer is defined in the actual ML code, in `lib/python/flame/optimizer` which can take in hyperparameters if any and should return the aggregated weights in either PyTorch of Tensorflow format. After implementation, the new optimizer needs to be registered into both `lib/python/flame/optimizer.py` and `lib/python/flame/config.py`.

### Currently Implemented Optimizers
1. FedAvg (i.e., weighted average in terms of dataset size)
```json
# e.g.
"optimizer": {
    "sort": "fedavg",
    "kwargs": {}
}
```
2. FedAdaGrad (i.e., server uses AdaGrad optimizer)
```json
"optimizer": {
    "sort": "fedadagrad",
    "kwargs": {
        "beta_1": 0,
        "eta": 0.1,
        "tau": 0.01
    }
}
```
3. FedAdam (i.e., server uses Adam optimizer)
```json
"optimizer": {
    "sort": "fedadam",
    "kwargs": {
        "beta_1": 0.9,
        "beta_2": 0.99,
        "eta": 0.01,
        "tau": 0.001
    }
}
```
4. FedYogi (i.e., servers use Yogi optimizer)
```json
"optimizer": {
    "sort": "fedyogi",
    "kwargs": {
        "beta_1": 0.9,
        "beta_2": 0.99,
        "eta": 0.01,
        "tau": 0.001
    }
}
```

