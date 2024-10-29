% gtversion 1.0
% guitar instrument definition

SR = 48000;

material_tab = [7850 2e11;19300 7.9e10; 19050 2.08e11]; % 1: steel, 2: gold, 3: uranium

notes = [-8 -3 2 7 11 16]';                             % fundamental, in semitones ± from middle C
material = [1 1 1 1 1 1]';                              % materials, referring to table
inharmonicity = [0.00001 0.00001 0.000001 0.000001 0.00001 0.00001]';   % inharmonicity---a small positive number!
L = 0.68;                                               % string length (common to all strings)
T60_0 = [15 15 15 15 15 15]';                           % T60 at DC for all strings
T60_1000 = [7 7 7 7 7 7]';                              % T60 at 1 kHz for all strings


string_def = string_def_gen(material_tab, notes, material, inharmonicity, L, T60_0, T60_1000);

string_num = 6;

out_num = 6;
output_def = [1 0.9; 2 0.9; 3 0.9; 4 0.9; 5 0.9; 6 0.9];
normalize_outs = 1;

pan = [0.314723686393179   0.405791937075619  -0.373013183706494   0.413375856139019  0.132359246225410  -0.402459595000590];

backboard = [-0.002 -0.001 -0.0002];
fretnum = 20;

frets = [0.123456789, -0.000987654321];
frets = fret_def_gen(frets, 20, 1, -0.001);

barrier_params_def = [1e10 1.3 10 20 1e-12];

itnum = 20;
tol = 1e-12;

finger_num = 6;
finger_params = [0.005 1e7 3.3 100];

