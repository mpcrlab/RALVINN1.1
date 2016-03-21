def valueFunction(state):
    if state[0] == 2 and state[1] == 0:
        return 100, 1
    return 0, 0

def threeColorValueFunction(state):
    if state[0] == 2 and state[1] == 0 and state[2] == 0:
        return 1, 1
    return 0, 0
