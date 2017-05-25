# Decision Train

Boosting-like algorithm

## C++ HowTo
Compile the testing executables
```bash
cmake .
make run_model run_tests
```

Run the test on toy data
```bash
./run_model dt_pid_model.json toy_test_dt_flat4d.csv toy_predictions
```

Run unit tests with
```bash
./run_tests
```

The code to dump `PIDEstimator(DecisionTrain)` to json is in `dump_pid_model.py`. The model is 
available at `man1-ipython01.cern.dev.yandex.net:/home/kazeevn/dt_pid_model.json`.