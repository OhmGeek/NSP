# NeuralStockPredictor
This doesn't work, not the code, the whole premise. Don't expect to get rich quick! The code behind this is a basic Feed Forward neural network, trained with RPROP, which I wrote from scratch. It doesn't have multithreading either, so not useful for most things.

## Why write this?
This was written as part of my EPQ, investigating whether one can predict Stock Market Indices using Feed Forward Networks, trained with Resilient Propagation. The answer is no, both found from a limited amount of experimental evidence (the program is only single threaded, so it takes a while to get data), as well as from reasoning obtained from many other experts in the field of Economics.

The code itself was written with Visual Studio 2013, in C#. It has been tested on Windows 7 and 8/8.1. It might also work on Linux/Mac if compiled with Mono, although that's not definite (I haven't tested it, and I probably won't).

## How to use:

Just build the project, and use the menu in the console program to navigate. One can load in a data set to train with, and then enter another set of data to test with. In addition, the user can enter values by hand and get an output regarding the next value. 

I hope you like this, it's not entirely useful but hopefully you find it useful.
