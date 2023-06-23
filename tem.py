train = [
    82.33,
    75.72,
    84.39,
    79.37,
    95.46,
    89.45,
    96.42,
    95.37,
]

test = [
    74.77,
    74.31,
    75.28,
    76.85,
    78.70,
    77.81,
    81.80,
    84.58,
]

for v_train, v_test in zip(train, test):
    print("{:.2f}".format(v_train - v_test))
