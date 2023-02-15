# What do we do?
## We are working on the mixed-integer nonlinear programming (MINLP) problem of distributing the data payload from lpGBTs (Low Power Gigabit Transceiver) to S-LINKs at the back-end part of HGCAL.

# On the way to assign lpGBT inputs to SLink outputs
## Basically:
- [`merge.ipynb`](merge.ipynb) merges module event data sizes with the hierarchy of modules per `lpGBT`, producing event data sizes per lpGBT (single or pair):
  - [`dat/rates`](dat/rates) contains a map of event sizes for each module.
  - [`dat/maps`](dat/maps) contains a description of the architecture that allows to link modules and `lpGBT`s.
- [assign.ipynb](assign.ipynb) then solves the NP-hard problem of distributing the `lpGBT`s over the `Slink`s
