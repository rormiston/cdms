# CDMS - Dark Matter Search
Machine learning code to estimate actual energy deposited into detector
based on the trace of multiple detectors on the crystal. 

## Before Running
1. In order to use these scripts, a few libraries need to be installed (keras, 
scipy etc.) so you can install them manually or run the `install.sh` file

```bash
$ chmod 755 install.sh
$ ./install.sh
```

This will create a virtual environment with the libraries install to it.

2. Make the mat file. The data is initially in a JSON file but it's easier
to manipulate an ndarray so you need to make and save a mat file. To do this, 
you can you the `make_data` function in `cdms_lib.py` as follows

```bash
$ python
>>> from cdms_lib import make_data
>>> json_data = 'cdms_data.json'
>>> make_data(json_data)
```

Then you can run the scripts as usual
```bash
$ python convnet.py
```

3. The `install.sh` file installs tensorflow-gpu, so you have to have access
to gpu cards in order to run it this way. If you don't have gpu cards, you can just do

```bash
$ pip uninstall tensorflow-gpu 
$ pip install tensorflow
```

**NOTE:** This may take quite a few cores and a lot of RAM. If you don't have
sufficient memory or CPUs, you'll get an ugly, albeit harmless,
"ResourceExhaustedError".
