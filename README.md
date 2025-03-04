# FEniCSx codes for finite viscoelasticity of elastomeric materials 

This repository contains a diverse array of example FEniCSx finite element simulation codes for a large deformation multi-branch viscoelasticity theory of elastomeric materials, including dynamic effects.

Specifically, the repository contains all the relevant FEniCSx Jupyter notebooks, mesh files, and experimental data files which were used in the representative example simulations in the accompanying paper, "A finite deformation viscoelasticity theory for elastomeric materials and its numerical implementation in the open source finite element program FEniCSx," by Eric M. Stewart and Lallit Anand. Find the paper [here](https://doi.org/10.1016/j.ijsolstr.2024.113023).

# Running the codes

A detailed guide for installing FEniCSx in a Docker container and running the notebooks using VSCode is provided in this repository, both for [Mac](https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FEniCSx_v08_Docker_install_mac.pdf) and [Windows](https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FEniCSx_v08_Docker_install_windows.pdf). The installation process is essentially similar for the two operating systems, but the example screenshots in the instructions are from the relevant system.

These are our preferred methods for editing and running FEniCSx codes, although [many other options exist](https://fenicsproject.org/download/). Note that all codes were written for FEniCSx v0.8.0, so our instructions documents will direct you to install this specific version of FEniCSx.

We have also provided a python script version of simulation FV09 which is meant to be run with MPI parallelization. To run this script in parallel on e.g. four cores use the following command syntax in the terminal:  

```
mpirun -n 4 python3 FV09_NBR_bushing_shear_MPI.py
```

# Movies

<br/><br/>

![](https://github.com/ericstewart36/finite_viscoelasticity/blob/main/example_movies.gif)

# Citation
E. M. Stewart and L. Anand. A large deformation viscoelasticity theory for elastomeric materials and its numerical implementation in the open-source finite element program FEniCSx. *International Journal of Solids and Structures*, 303:113023, Oct. 2024.
