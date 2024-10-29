# Multi Plate 3D Information

This document describes the instrument and score file formats for the Multi Plate 3D instrument type. This implements a set of non-linear plates or membranes embedded within a 3D airbox. It can also implement a bass drum.

## Instrument File Format

The instrument file format is similar to Zero Code. A sample instrument file is given below:

```
# mpversion 0.1
samplerate 44100
airbox 1.32 1.32 1.37 340.0 1.21
plate plat1 0.81 0.87 0.0 0.0 0.22 7800.0 0.002 8e11 0.33 4.0 0.001
plate plat2 0.39 0.42 -0.1 -0.1 0.0 7800.0 0.002 8e11 0.33 4.0 0.001
plate plat3 0.65 0.61 0.1 0.1 -0.27 7800.0 0.002 8e11 0.33 4.0 0.001
airbox_output 0.01 0.02 0.6
airbox_output -0.6 0.012 0.15
airbox_output -0.6 0.012 -0.15
airbox_output 0.01 0.02 -0.6
airbox_output 0.01 0.7 0.01
airbox_output -0.01 -0.7 -0.01
airbox_output 0.6 0.012 0.15
airbox_output 0.6 0.012 -0.15
plate_output plat1 0.141421356237310 0.113137084989848
plate_output plat1 -0.282842712474619 0.056568542494924
plate_output plat2 0.141421356237310 0.113137084989848
plate_output plat2 -0.282842712474619 0.056568542494924
plate_output plat3 0.141421356237310 0.113137084989848
plate_output plat3 -0.282842712474619 0.056568542494924
```

The following lines are allowed in the instrument file:

- `samplerate` sets the sample rate for the simulation in Hz. Default is 44100.
- `airbox` defines the dimensions and other parameters of the airbox. Parameters are the width, depth, height, `c_a` and `rho_a`. Only one airbox can be defined.
- `plate` defines a plate within the airbox. The first parameter is a name for the plate which must be a unique string and is used to refer to it for the purposes of outputs and strikes. The numeric parameters are size X, size Y, centre X, centre Y, centre Z, `rho`, `H`, `E`, `nu`, `T60` and `sig1`.
- `airbox_output` defines an output taken from within the airbox. The parameters are its X, Y and Z position.
- `plate_output` defines an output taken from a plate. The parameters are the name of the plate and the X and Y position for the output. The position values are normalised to the range -1 to +1.
- `bassdrum` defines a bass drum embedded in an airbox. This should be used instead of having separate airbox and plate lines. The parameters are: airbox width, airbox depth, airbox height, `c_a`, `rho_a`, drum shell height, drum radius, membrane rho, `H`, `T`, `E`, `nu`, `T60`, `sig1`. For the purposes of adding strikes and taking outputs, the top membrane is named `drumtop` and the bottom one `drumbottom`.

Note that items must come after any items that they are dependent on: the airbox must be defined before the plates, and the plates should be defined before the outputs.


## Score File Format

The score file defines the length of the simulation in seconds, and also any strike events that should happen. Example score file:

```
duration 4.0
strike 2.2675737e-4 plat1 0.6653034 0.3507947 0.007233560090703 1000.0
strike 0.22684807256236 plat2 0.5177853 0.41139928754 0.006507936507936508 500.0
strike 0.4546031746031746 plat3 0.68823711 0.363045233 0.00780045351473923 500.0
```

The first parameter of a strike is the start time. The other parameters are the name of the plate, the X position, the Y position, the duration, and the maximum force. The position values are normalised to the range 0-1 and the duration and start time are given in seconds.
