# Flame SDK

## Selector

### Currently Implemented Selectors
1. Naive (i.e., select all)
2. Random (i.e, select k out of n local trainers)

Users are able to implement new selectors in `lib/python/flame/selector/` which should return a dictionary with keys corresponding to the active trainer IDs (i.e., agent IDs). After implementation, the new selector needs to be registered into both `lib/python/flame/selectors.py` and `lib/python/flame/config.py`. 
