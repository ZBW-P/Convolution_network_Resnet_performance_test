# Convolution_network_Resnet_performance_test
Running resnet in differenrt device environment test performance

### How to use C1, just run the below python code:
#### For the following code without using --profiler allows the code to run normal version of counting by using time.perf counter(), if add the --profiler we can ask the profiler to record the time.


python C1.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C1.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler


### How to use C2, just run the below python code:
#### For the following code without using --profiler allows the code to run normal version of counting by using time.perf counter(), if add the --profiler we can ask profiler to record the time.


python C2.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C2.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler


### How to use C3, just run the below python code:
#### For the following code without using --profiler allows the code to run normal version of counting by using time.perf counter(), if add the --profiler we can ask the profiler to record the time.


python C3.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C3.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler


### How to use C4, just run the below python code:
#### For the following code without using --profiler allows the code to run normal version of counting by using time.perf counter(), if add the --profiler we can ask the profiler to record the time. If we run with --num_workers 'number' we are able to adjust the number of workers.


python C4.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C4.py --data_path ./data --num_workers 1 --batch_size 128 --optimizer sgd --epochs 5
python C4.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler
python C4.py --data_path ./data --num_workers 1 --batch_size 128 --optimizer sgd --epochs 5 --profiler


### How to use C5, just run the below python code:
#### For the following code without using --profiler allows the code to run normal version of counting by using time.perf counter(), if add the --profiler we can ask the profiler to record the time. If we run with --cuda the running would adjust device to gpu


python C5.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C5.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C5.py --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler
python C5.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5 --profiler


### How to use C6, just run the below python code:
#### For all the following commands need a gpu and the different name of optimizer can run different optimizer methods, adjust epoch can run more epochs.


python C6.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5
python C6.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer SGD --epochs 5
python C6.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer Adagrad --epochs 5
python C6.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer Adadelta --epochs 5
python C6.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer Adam --epochs 5


### How to use C7, just run the below python code


python C7.py --cuda --data_path ./data --num_workers 0 --batch_size 128 --optimizer sgd --epochs 5


### How to use Log1~Log5 series to see profiler
#### TO Enable debug pages
chrome://chrome-urls
#### Use the tracing and the json in dictionary Log 1~5 to see the profiler record
chrome://tracing/
