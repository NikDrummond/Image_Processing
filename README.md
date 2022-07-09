## Image_Processing

Some quick and simple tools for image pre-processing and getting some metrics for the Borst lab

Installation:

pip and git are both needed, to install these on mac use:

```
conda install git pip
```

OpenCV is required, so to install this on mac open the command line and use:

First of all try:

```conda install opencv
```

If this doesnt work try:

```conda install -c conda-forge opencv
conda install -c conda-forge/label/gcc7 opencv
conda install -c conda-forge/label/broken opencv
conda install -c conda-forge/label/cf201901 opencv
conda install -c conda-forge/label/cf202003 opencv 
```

If this still doesn't work, panic and let me know

To then install the Image Processing toolbox:

```
pip install git+https://github.com/nikdrummond/Image_Processing/
```

See [example](examples.ipynb) for a walk through
