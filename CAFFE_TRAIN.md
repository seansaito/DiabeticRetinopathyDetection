# Training using CAFFE

## Preparation

* Make sure you have caffe installed on your machine

## Generating the dataset

* Caffe uses LMDB, which is a persistent data store (basically a memory/storage-efficient
database) to keep data. This way, we don't have to load all examples into memory and
potentially crash the computer.

* Change the directories in lmdb_generator.py (the lines located at the top of the
script) to point to the training image, training labels, testing images, and teesting labels.

* Run `python lmdb_generator.py`. This will generate two lmdbs, namely
`max_cropped_rotated_train_lmdb` and `max_cropped_rotated_test_lmdb`

## Training the network

* Make sure that the `.prototxt` files are in the same directory as `lmdb_generator.py`.

* Make a directory called `results`. This is where the model will be saved (with the
extension `.caffemodel`)

* If you are using a remote machine, it's a good idea to use `tmux` or `screen` so even 
if you log-off, the program will still be running

* For tmux, put the following lines: (assuming you are in the same directory as `lmdb_generator.py`)

```
$ tmux attach
$ python
>>> import caffe
>>> net = caffe.get_solver("alexnet_solver.prototxt")
>>> ... # A bunch of output
>>> net.solve()
```

This will start the training.

## Inference / Evaluation

* The Jupyter Notebook `Retinopathy Inference.ipynb` gives you everything you need.
Just follow the instructions.
