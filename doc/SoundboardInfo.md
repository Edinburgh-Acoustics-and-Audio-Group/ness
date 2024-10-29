# Soundboard Information

The soundboard model simulates a soundboard with a number of strings attached, as well as barriers (frets). This document describes the instrument and score file formats for this model.

**Warning: there is a possible unresolved numeric instability bug affecting this instrument. You may need to experiment with different settings to avoid problems.**

## Instrument File Format

The instrument file format defines the configuration of the strings, soundboard and frets. A sample instrument file, corresponding to the configuration in the original Matlab, is given below:

```
samplerate 44100
string string1 0.662944737278636 0.020632359246225 79.150136708685949 2.191433389648589e+11 0.000384352256525255 11.311481398313173 8.357470309715547 0.655477890177557 0.276922984960890 0.694828622975817 0.438744359656398 3 -0.001418729661716 -0.002689803992052 0.019597439585161
string string2 0.681158387415124 0.020097540404999 79.297770703985535 2.097075129744568e+11 0.000483147105037813 10.071423357148380 8.515480261156666 0.171186687811562 0.046171390631154 0.317099480060861 0.381558457093008 9 -0.001509373363965 -0.003674776529611 0.013403857266661
string string3 0.525397363258701 0.020278498218867 63.152261633550964 2.160056093777760e+11 0.000458441465911911 11.698258611737554 8.486264936249832 0.706046088019609 0.097131781235848 0.950222048838355 0.765516788149002 8 -0.000552050153997 -0.003762004636883 0.015852677509798
string string4 0.682675171227804 0.020546881519205 79.411855635212305 2.028377267725443e+11 0.000491898485278581 11.867986495515101 7.784454039068336 0.031832846377421 0.823457828327293 0.034446080502909 0.795199901137063 12 -0.001359405353707 -0.003003271896036 0.012238119394911
plate 7850 0.001 2e11 0.3 0 0.5 0.2 10 9
collision 1.793912177273139e+15 1.555095115459269 50
string_out string3 0.585264091152724 0.514400458221443
string_out string2 0.549723608291140 0.507458188556991
string_out string2 0.917193663829810 -0.239108306049287
string_out string4 0.285839018820374 0.135643281450442
plate_out 0.075854289563064 0.779167230102011 0.137647321744385
plate_out 0.053950118666607 0.934010684229183 -0.061218717883588
plate_out 0.530797553008973 0.129906208473730 -0.976195860997517
```

The following lines are allowed in the instrument file:

- `samplerate` sets the sample rate for the simulation. Default is 44100.
- `string` defines a string within the simulation. The first parameter is a name for the string which must be unique and is used to refer to it for the purposes of inputs and outputs. The numeric parameters are length (m), density (kg/m), tension (N), Young's modulus (Pa), radius (m), T60 at DC (s), T60 at 1kHz (s), X co-ordinate of start of string, Y co-ordinate of start of string, X co-ordinate of end of string, Y co-ordinate of end of string, number of frets, fret height (m, should be negative!), baseboard height (m, should be negative and less than fret height), variation in baseboard profile (quadratic, small positive number).
- `plate` defines the soundboard to which the strings are connected. There should only be one plate per instrument file. The numeric parameters are density (kg per cubic metre), thickness (m), Young's modulus (Pa), Poisson's ratio (nd), tension (N/m), size in X dimension (m), size in Y dimension (m), T60 at DC (s), T60 at 1kHz (s).
- `collision` defines parameters for the Newton Raphson solver that handles collisions between the strings and the frets. They are: collision stiffness, collision nonlinearity exponent, and number of iterations.
- `string_out` defines an output taken directly from a string. Parameters are name of string, location on string (normalised to range 0 to 1), and pan position for stereo mix (normalised to range -1 to 1).
- `plate_out` defines an output taken directly from the soundboard. Parameters are X position (0 to 1), Y position (0 to 1), and stereo pan position (-1 to 1).

**Order is important**: the strings must come first, followed by the plate definition, followed by the collision definition, and finally the outputs.


## Score File Format

The score file defines input events that happen during the simulation. It also defines the length of the simulation in seconds. The soundboard model supports two types of inputs: striking a string, and plucking a string. Example score file:

```
duration 2.0
strike string1 0.001000000000000 0.001505957051665 5.736077649819015 0.929263623187228
pluck string2 0.002000000000000 0.001699076722657 3.693122214143396 0.349983765984809
strike string3 0.003000000000000 0.001890903252536 3.746470027795287 0.196595250431208
pluck string4 0.004000000000000 0.001959291425205 4.287541270618682 0.251083857976031
```

The first parameter of a strike or pluck is the name of the string which is to be struck or plucked. The numeric parameters are: start time (seconds), duration (seconds), maximum force (N), and location on string (normalised to range 0 to 1).
