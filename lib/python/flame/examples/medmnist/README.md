## MedMNIST

We will use the PathMNIST dataset from (MedMNIST)[https://medmnist.com/] to go over an example of using an adaptive aggregator on a data heterogeneity setting.
This example is run using `conda`.
If the environment is not set up, you may follow the instructions for setup [here](../../../../../docs/08-flame-sdk.md).
In the `medmnist` directory, we start by using the command `conda activate flame`.
This needs to be done before the shell script is run in a terminal.

It is also important to change the job ID to a new random job ID before running this (or any other) example while using the mqtt broker.
If anyone else is running the example with the same job ID (and broker), the program may behave in unexpected ways (for instance, one of the trainers may correspond with a different aggregator that was not intended for it).
The ID should be changed in `aggregator/fedavg.json`, `trainer/fedavg/fedavg1.json`, `trainer/fedavg/fedavg2.json`, and `trainer/fedavg/fedavg3.json`.
Make sure you changed the job IDs to the same (new) job ID in all four files.
The task IDs should remain the same as before.

Once you are back in the `medmnist` directory, you can run three trainers along with one aggregator with `bash example.sh fedavg pytorch`.

By using `cat example.sh` we can see how this file sets up the federated learning.

```
cd trainer/$1

for i in {1..3}
do
    rm -rf $1$i
    mkdir $1$i
    cd $1$i
    python ../../$2/main.py ../$1$i.json > log$i.txt &
    cd ..
done

cd ../../aggregator
rm -f $1_log.txt
python $2/main.py $1.json > $1_log.txt &
```

The first parameter specifies the optimizer, and the second parameter specificies the framework used.

Initially, we run three trainers in different working directories (this keeps the downloaded files seperate) using three different config files `fedavg1.json`, `fedavg2.json`, and `fedavg3.json`.
These files are located under the `fedavg` folder.
The log file for trainer `i` is located under `fedavg/fedavg$i`.
After this, the script runs the pytorch aggregator along with its configuration within `aggregator`.
The log file for this will be under `aggregator` as well, and is named `fedavg_log.txt`.
In order to check the progress of the program you may run `cat trainer/fedavg/fedavg1/log1.txt` from the `medmnist` folder.
This will look at the output of the first trainer until then.

## Keras

In order to use keras to run these examples, we need to change change the second argument for the script. For the example above, we would run `bash example.sh fedavg keras`.

## Other Optimizers

Config files for other optimizers have been created as well.
To test out different optimizers, you may run the script by changing the first parameter to another optimizer. Optimizers currently available for this example are `fedavg`, `fedadagrad`, `fedadam`, and `fedyogi`.

Keep in mind that the config files you may need to change are in two different locations (as discussed earlier), and please make sure to change the job IDs provided to avoid collisions.
