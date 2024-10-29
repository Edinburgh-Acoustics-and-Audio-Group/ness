# Zero Code Information

The Zero Code was the first instrument model to be made available through NESS. It consists of a collection of 2D linear plates with connections between them. This document describes the instrument and score file formats used for this code.


## Instrument File Format

Each line of the instrument file specifies either a component, a connection, an output, or the sampling rate. The only supported component type is `plate`, a 2D rectangular plate. The first word on the line must be either `plate`, `connection`, `output` or `samplerate` and it is followed by a number of parameters specified by spaces. Blank lines are ignored. Comments (beginning with `#`) are allowed.

Parameters for each entry type:

- `plate <name> <material name> <thickness> <tension> <X size> <Y size> <T60 at 0Hz> <T60 at 1000Hz> <boundary condition type>`
- `connection <component 1> <component 2> <x1> <y1> <x2> <y2> <linear stiffness> <nonlinear stiffness> <T60>`
- `output <component> <x> <y> <pan>`
- `samplerate <sampling rate in Hz>`

Notes:

- Components are each assigned a name which can be any string value as long as it is unique and doesn't contain spaces. This name is then used to refer to the component for `connection` and `output` definitions, and in the score file.
- Material names are looked up in the `materials.txt` file to get parameters for that material.
- Boundary condition types are: 1=clamped, 2=simply supported, 3=free, 4=free with fixed corners.
- x and y co-ordinates on the components are normalised to the range 0-1, regardless of the actual size of the component.
- Pan value for output specifies where it goes in the stereo mix in the range -1 to 1 (where -1=left, 0=centred, 1=right).
- The default sample rate if not given is 44100Hz.

Example instrument file:

```
# set 44100Hz sampling rate
samplerate 44100

# define a steel plate
plate plat1 steel 0.002 0.0 0.3 0.2 10.0 6.0 4  

# define a connection from one point on the plate to another
connection plat1 plat1 0.8 0.4 0.6 0.7 10000.0 10000000.0 1000000.0

# define two outputs from the plate
output plat1 0.9 0.6 -1.0
output plat1 0.3 0.7 1.0
```


## Score File Format

The score file defines inputs that happen at certain times during the simulation. The format is similar to the instrument file format; each line specifies either an input (`bow` and `strike` are supported), the simulation duration, and the high pass filter state. Again blank lines are ignored and comments are allowed. Parameters are as follows:

- `strike <start time> <component> <x> <y> <duration> <amplitude>`
- `bow <start time> <component> <x> <y> <duration> <force amplitude> <velocity amplitude> <friction> <ramp time>`
- `duration <duration>`
- `highpass [on|off]`

Notes:

- Component names and locations within components are as described in the instrument file section.
- Durations and start times are in seconds, not samples.
- Ramp time is the time taken for the bow velocity and force to reach its peak amplitude.

Example score file:

```
highpass off   # no high-pass filter
duration 1.0   # one second simulation

# define a strike
strike 0.0 plat1 0.4 0.7 0.002 400000.0

# define a bowing action
bow 0.3 plat1 0.3 0.9 4.0 2.3 2.8 1.1 0.02
```


## Materials File Format

The materials file should be named `materials.txt` and should be in the current directory when the code is run. It is a text file where each line specifies the parameters for one material:

- `<material name> <young's modulus> <poisson's ratio> <density>`

The material name can be any unique string that doesn't contain spaces. Again blank lines are ignored and comments are allowed.

Example materials file:

```
# Materials list
steel 2e11 0.3 7800
aluminium 7e10 0.35 2700
lead 1.6e10 0.44 11340
```


## Zero Point 1 Code

A revision of the Zero Code, called the Zero Point 1 Code, was later made available. This is nearly identical to the Zero Code but has two additional features:

Firstly, in the instrument file, connections are defined differently, giving more flexibility when creating them. The new syntax is as follows:

- `connection <component 1> <component 2> <x1> <y1> <x2> <y2> <stiffness> <nonlinearity exponent> <one sided> <offset>`

Example Zero Point 1 instrument file:

```
# set 44100Hz sampling rate
samplerate 44100

# define a steel plate
plate plat1 steel 0.002 0.0 0.3 0.2 10.0 6.0 4 

# define a connection from one point on the plate to another
connection plat1 plat1 0.8 0.4 0.6 0.7 100000.0 1 1 0.653

# define two outputs from the plate
output plat1 0.9 0.6 -1.0
output plat1 0.3 0.7 1.0
```

Secondly, in the score file, an audio file can be used as an input. This will cause the plate to be displaced according to the audio file selected, which must be a mono 8- or 16-bit PCM WAV file:

- `audio <filename> <start time> <component> <x> <y> <gain>`

Example Zero Point 1 score file:

```
highpass off   # no high-pass filter
duration 1.0   # one second simulation

# define a strike 
strike 0.0 plat1 0.4 0.7 0.002 400000.0

# define a bowing action
bow 0.3 plat1 0.3 0.9 4.0 2.3 2.8 1.1 0.02

# define an audio input
audio drumming.wav 0.1 plat1 0.2 0.4 1.0
```
