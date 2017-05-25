# Decision Train

Boosting-like algorithm

## C++ HowTo
Compile the model testing executable
```bash
g++ -Wall -Werror dt_pid.cpp dtrain.cpp test_model.cpp -o test_model -std=c++17 -O2
```

Run the test on toy data
```bash
./test_model dt_pid_model.json toy_test_dt_flat4d.csv toy_predictions
```

The code to dump `PIDEstimator(DecisionTrain)` to json is in `dump_pid_model.py`. The model is 
available at `man1-ipython01.cern.dev.yandex.net:/home/kazeevn/dt_pid_model.json`.