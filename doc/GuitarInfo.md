# Guitar Information

This document describes the instrument and score file formats for the guitar model. For reasons of compatibility with the original Matlab version of this code, these files use a subset of Matlab syntax.


## Instrument File Format

Values that can be defined in the instrument file are described below:

- **`SR` (scalar)** - sampling rate for the simulation. Defaults to 44100 if not specified.
- **`string_def` (array)** - defines the parameters for each string of the guitar. This is a 2-dimensional array with a row for each string. Each row contains 7 items: length in metres, Young's modulus, tension, radius, density, T60 at 0Hz, T60 at 1000Hz.
- **`output_def` (array)** - defines the locations of the outputs. This is a 2-dimensional array with a row for each output. Each row contains 2 items: the index of the string from which the output should be taken (1-based), and the distance along the string (normalised to the range 0-1).
- **`pan` (array)** - defines the pan position for each output in the stereo mix. This is a 1-dimensional array with a single value (range -1 to 1) for each output.
- **`backboard` (array)** - defines the shape of the backboard. This is a 3 element array; the elements (which should be negative) define a quadratic function describing the shape of the backboard.
- **`fretnum` (scalar)** - the number of frets. Will be computed automatically from the size of the `frets` array if not specified.
- **`frets` (array)** - defines the locations of the frets. This is a 2-dimensional array with a row for each fret. Each row contains 2 items: the position of the fret (normalised to the range 0-1) and the height of the fret (which should be negative).
- **`barrier_params_def` (array)** - specifies 5 basic parameters for the barrier (fret and backboard) collisions. The parameters are: K, alpha, beta, number of iterations for Newton solver, and tolerance for Newton solver.
- **`finger_params` (array)** - specifies 4 basic parameters for the finger collisions. The parameters are: mass, K, alpha and beta.
- **`normalize_outs` (scalar)** - set this to 1 to normalise the output arrays individually before mixing, or 0 to leave them as they are.

Example instrument file:

```
SR = 48000;

string_def = [0.68 2e11 12.1 0.0002 7850 15 5;
              0.68 2e11 12.3 0.00015 7850 15 5;
              0.68 2e11 21.9 0.00015 7850 15 5;
              0.68 2e11 39.2 0.00015 7850 15 7;
              0.68 2e11 27.6 0.0001 7850 15 5;
              0.68 2e11 49.2 0.0001 7850 15 8];

output_def = [1 0.9; 2 0.9; 3 0.9; 4 0.9; 5 0.9; 6 0.9];

pan = [0.3 0.4 -0.4 0.4 0.1 -0.4 ];

backboard = [-0.002 -0.001 -0.0002];

fretnum = 20;
frets = [0.056125687318306  -0.001000000000000;
         0.109101281859661  -0.001000000000000;
         0.159103584746285  -0.001000000000000;
         0.206299474015900  -0.001000000000000;
         0.250846461561659  -0.001000000000000;
         0.292893218813453  -0.001000000000000;
         0.332580072914983  -0.001000000000000;
         0.370039475052563  -0.001000000000000;
         0.405396442498639  -0.001000000000000;
         0.438768975845313  -0.001000000000000;
         0.470268452820352  -0.001000000000000;
         0.500000000000000  -0.001000000000000;
         0.528062843659153  -0.001000000000000;
         0.554550640929830  -0.001000000000000;
         0.579551792373143  -0.001000000000000;
         0.603149737007950  -0.001000000000000;
         0.625423230780830  -0.001000000000000;
         0.646446609406726  -0.001000000000000;
         0.666290036457491  -0.001000000000000;
         0.685019737526282  -0.001000000000000];
         
barrier_params_def = [1e10 1.3 10 20 1e-12];

finger_params = [0.005 1e7 3.3 100];
```

To simplify the specification of the string and fret information, two functions are provided: `string_def_gen` and `fret_def_gen`. These take parameters that are more musically meaningful and return the `string_def` and `fret_def` arrays as required by the code.

The `string_def_gen` function takes 7 parameters:

- `material_tab` - This 2D array specifies materials that can then be used to create the strings. Each row of the array corresponds to a separate material, and should contain the density and Young's Modulus parameters for that material.
- `notes` - This array contains one element per string, and specifies the pitch of the string, in semitones relative to middle C.
- `materials` - This array also contains one element per string. Each element is an index into the `material_tab` above, specifying the material for the string.
- `inharmonicity` - This array also contains one element per string, which should be a small, positive number.
- `length` - This is a scalar value and specifies the length of the strings in metres.
- `T60_0` - This array contains one element per string, specifying the T60 at d.c. for each string.
- `T60_1000` - Same as `T60_0`, but specifies the T60 at 1000Hz for each string.

The `fret_def_gen` function generates an array of frets that are exponentially spaced (i.e. evenly spaced in terms of pitch). It takes 3 or 4 parameters:

- `frets` - The first parameter is an optional array of manually positioned frets. If this is passed in, the new frets generated by the function will be appended onto it.
- `fretnum` - The number of frets to generate (usually 20-24).
- `wid` - The spacing of the frets in semitones (usually 1).
- `height` - The height of the frets, normally around -0.001 (around a millimetre below the string).

Example instrument file using the functions:

```
SR = 48000;

% 1 - steel, 2 - gold, 3 - uranium
mat_tab = [7850 2e11;19300 7.9e10;
           19050 2.08e11];

% fundamental, in semitones relative to middle C
notes = [-8 -3 2 7 11 16];

% materials, referring to table
material = [1 1 1 1 1 1];

% inharmonicity---a small positive number!
inharmonicity = [0.00001 0.00001 0.000001 0.000001 0.00001 0.00001];

% string length (common to all strings)
L = 0.68;

% T60 at DC for all strings
T60_0 = [15 15 15 15 15 15];

% T60 at 1 kHz for all strings
T60_1000 = [7 7 7 7 7 7];

string_def = string_def_gen(mat_tab, notes, material, inharmonicity, L, T60_0, T60_1000);

out_num = 6;
output_def = [1 0.9; 2 0.9; 3 0.9; 4 0.9; 5 0.9; 6 0.9];
normalize_outs = 1;

pan = [0.3 0.4 -0.4 0.4 0.1 -0.4];

backboard = [-0.002 -0.001 -0.0002];

frets = [0.123456789, -0.000987654321];
frets = fret_def_gen(frets, 20, 1, -0.001);

barrier_params_def = [1e10 1.3 10 20 1e-12];

finger_params = [0.005 1e7 3.3 100];
```

## Score File Format

The score file for the guitar code is also in Matlab format. This time there are
only three values allowed:

- **`Tf` (scalar)** - specifies the duration of the simulation in seconds.
- **`exc` (array)** - defines all of the excitations. This is a 2-dimensional array. Each row represents a single half-sinusoidal pluck of one of the strings, and consists of 5 values: the index of the string, the start time in seconds, the position on the string (normalised to the range 0-1), the duration in seconds, and the amplitude.
- **`finger_def` (cell array)** - defines all of the fingers in the simulation, their movements and the forces associated with them. This is a 2-dimensional cell array. Each row represents one finger and consists of 3 elements: the index of the string that the finger is on; a 2-dimensional array defining how the finger position and force changes over time; and a two element array specifying the finger's initial position and velocity. Each row of the middle element contains a time (in seconds), a position and a force. The position and force are interpolated between the times given.

Example score file:

```
% guitar score file

Tf = 1;           % duration

exc = [ 1   0.010000000000000   0.8   0.001299965854261   0.753352341821251;
	    2   0.022500000000000   0.8   0.001734179906576	  0.570954585011654;
	    3   0.035000000000000   0.8   0.001104209253757   1.125803331171040;
	    4   0.047500000000000   0.8   0.001792575487554   0.524681470128999;
	    5   0.060000000000000   0.8   0.001782728942752   0.562042492509529;
	    6   0.072500000000000   0.8   0.001532397693219   0.629611982278315;
	    6   0.260000000000000   0.8   0.001450614072287   1.141631167720519;
	    5   0.272500000000000   0.8   0.001672335843923   1.286386643614576;
	    4   0.285000000000000   0.8   0.001856110833478   0.789150493176199;
	    3   0.297500000000000   0.8   0.001498445424255   0.997868010687049;
	    2   0.310000000000000   0.8   0.001048784504036   1.318434615976952;
	    1   0.322500000000000   0.8   0.001313832359873   1.095128292529225];

finger_def = {
    1, [0 0.038 0; 0.18 0.148 0; 0.31 0.093 1.0; 0.78 0.173 1.0; 1.0 0.283 1.0], [0.01, 0];
    2, [0 0.236 0; 0.18 0.153 0; 0.31 0.168 1.0; 0.78 0.205 1.0; 1.0 0.027 1.0], [0.01, 0];
    3, [0 0.157 0; 0.18 0.195 0; 0.31 0.115 1.0; 0.78 0.194 1.0; 1.0 0.228 1.0], [0.01, 0];
    4, [0 0.250 0; 0.18 0.081 0; 0.31 0.120 1.0; 0.78 0.166 1.0; 1.0 0.133 1.0], [0.01, 0];
    5, [0 0.156 0; 0.18 0.272 0; 0.31 0.114 1.0; 0.78 0.265 1.0; 1.0 0.076 1.0], [0.01, 0];
    6, [0 0.064 0; 0.18 0.001 0; 0.31 0.264 1.0; 0.78 0.070 1.0; 1.0 0.073 1.0], [0.01, 0]
};
```

As with the instrument file, some functions are provided to assist in specifying the score. These are: `pluck_gen`, `strum_gen`, `strum_gen_multi` and `cluster_gen`.

`pluck_gen` is the simplest of the functions and simply appends a single pluck to the excitation array. It takes 6 parameters and returns the excitation array with the new pluck added. The parameters are:

- `oldexc` (array) - The existing excitation array.
- `string` (scalar) - The index of the string to be plucked (1-based).
- `T` (scalar) - The start time for the pluck in seconds.
- `dur` (scalar) - The duration of the pluck in seconds.
- `amp` (scalar) - The amplitude of the pluck in Newtons.

`strum_gen` generates a strum gesture, consisting of plucks on each string in order. It takes 11 parameters:

- `oldexc` (array) - The existing excitation array.
- `T` (scalar) - The start time of the strum.
- `dur` (scalar) - The duration of the entire strum gesture.
- `ud` (scalar) - The direction of the strum: 0 means up (starting on string 1), 1 means down (starting on last string).
- `amp` (scalar) - The amplitude in Newtons (normally around 1-10N).
- `amp_rand` (scalar) - Amount of random variation in amplitude (0-1).
- `pluckdur` (scalar) - The duration of each individual pluck in the gesture (normally around 0.001-0.01s).
- `pluckdur_rand` (scalar) - Amount of random variation in pluck duration (0-1).
- `pos` (scalar) - Average position on strings (0-1).
- `pos_rand` (scalar) - Amount of random variation in position (0-1).
- `times_rand` (scalar) - Amount of random variation in pluck times (0-1).

`strum_gen_multi` is similar, but generates a sequence of multiple strum gestures. It takes 12 parameters:

- `oldexc` (array) The existing excitation array.
- `T` (array) Array of times, spanning the length of the gesture.
- `density` (array) Array of densities, corresponding to the times in T.
- `dur` (array) Array of durations for strums, corresponding to the times in T.
- `ud` (scalar) Strum direction. 0 means up, 1 means down, 2 means alternating.
- `amp` (array) Array of amplitudes, corresponding to the times in T.
- `amp_rand` (scalar) Amount of random variation in amplitudes (0-1).
- `pluckdur` (array) Array of durations for individual plucks, corresponding to the times in T.
- `pluckdur_rand` (scalar) Amount of random variation in pluck durations (0-1).
- `pos` (array) Array of positions on strings, corresponding to the times in T (0-1).
- `pos_rand` (scalar) Amount of random variation in positions (0-1).
- `times_rand` (scalar) Amount of random variation in pluck start times (0-1).

`cluster_gen` generates a random cluster of plucks on all the strings. It takes 10 parameters:

- `oldexc` (array) The existing excitation array.
- `T` (scalar) Start time for the cluster.
- `N_pluck` (scalar) Total number of plucks in cluster.
- `dur` (scalar) Duration of entire gesture.
- `amp` (scalar) Amplitude in Newtons.
- `amp_rand` (scalar) Amount of random variation in amplitude (0-1).
- `pluckdur` (scalar) Duration of each individual pluck.
- `pluckdur_rand` (scalar) Amount of random variation in pluck duration (0-1).
- `pos (scalar) Average position on strings (0-1).
- `pos_rand` (scalar) Amount of random variation in position (0-1).

Example score file using the functions:

```
% guitar score file

Tf = 1;           % duration

exc = strum_gen(exc, 0, 0.4, 1, 1, 0.1, 0.001, 0.1, 0.8, 0.1, 0.1);
exc = strum_gen(exc, 0.3, 0.4, 0, 1, 0.1, 0.001, 0.1, 0.8, 0.1, 0.1);
exc = cluster_gen(exc, 0.6, 20, 0.2, 1, 0.1, 0.002, 0.1, 0.8, 0.1);
exc = pluck_gen(exc, 1, 0.9, 0.8, 0.001, 1);

T = [0, 0.2, 0.4, 0.6];
density = [5, 5, 5, 5];
dur = [0.1, 0.1, 0.1, 0.1];
amp = [1, 1, 1, 1];
pluckdur = [0.001, 0.001, 0.001, 0.001];
pos = [0.8, 0.8, 0.8, 0.8];

exc = strum_gen_multi(exc, T, density, dur, 2, amp, 0.2, pluckdur, 0.2, pos, 0.2, 0.2);

finger_def = {
    1, [0 0.038 0; 0.18 0.148 0; 0.31 0.093 1.0; 0.78 0.173 1.0; 1.0 0.283 1.0], [0.01, 0];
    2, [0 0.236 0; 0.18 0.153 0; 0.31 0.168 1.0; 0.78 0.205 1.0; 1.0 0.027 1.0], [0.01, 0];
    3, [0 0.157 0; 0.18 0.195 0; 0.31 0.115 1.0; 0.78 0.194 1.0; 1.0 0.228 1.0], [0.01, 0];
    4, [0 0.250 0; 0.18 0.081 0; 0.31 0.120 1.0; 0.78 0.166 1.0; 1.0 0.133 1.0], [0.01, 0];
    5, [0 0.156 0; 0.18 0.272 0; 0.31 0.114 1.0; 0.78 0.265 1.0; 1.0 0.076 1.0], [0.01, 0];
    6, [0 0.064 0; 0.18 0.001 0; 0.31 0.264 1.0; 0.78 0.070 1.0; 1.0 0.073 1.0], [0.01, 0]
};
```

The excitation array, `exc`, is automatically initialised as an empty array, so it is not necessary to declare it before passing it to the first function.
