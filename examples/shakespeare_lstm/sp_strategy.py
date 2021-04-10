
from flwr.common.switchpoint import AccuracyVariance

def get_sp_strategy(s, is_server):
    SERVER_ACC_VAR = 1000
    CLIENT_ACC_VAR = 2000
    sp_strategy = None

    if s == SERVER_ACC_VAR:
        if is_server:
            sp_strategy = AccuracyVariance(3, 0.001, False)
    elif s == CLIENT_ACC_VAR:
        if not is_server:
            sp_strategy = AccuracyVariance(5, 0.001, False)

    return sp_strategy
