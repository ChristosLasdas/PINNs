# Physics-Informed Neural Networks (PINNs) and applications

In this repository there are Python codes written by Christos Lasdas in the context of the Master's thesis *"Modeling and Simulating Fluid Flows using Physics-Informed Neural Networks and Computational Fluid Dynamics"*.

One may find codes that tackle the following problems:

* *Steady State Flow in a Pipe-Channel (Laminar and Turbulent)*
* *Steady State Flow in a Lid Driven Cavity (Laminar and Turbulent)*
* *Transient State Flow in a Pipe-Channel*
* *Transient Flow in a Lid Driven Cavity*
* *Train the PINN for different Reynolds numbers and ask of it to predict the fluid flow for a different Reynolds number (intepolation and extrapolation) - Pipe Flow and Lid Driven Cavity flow*
* *Steady-State Flow around a Cylinder*
* *Transient Flow around a Cylinder*
* *Steady-State Flow around an Airfoil*

> These codes use the *DeepXDE* package, so an installation is to be done prior to running the code.
> We advise you to use GPU and TPU to run these codes as they require several hours or even several days to run in a CPU environment.
> If the codes do not run, there might be an issue with the versions of the packages and the libraries used in them.
