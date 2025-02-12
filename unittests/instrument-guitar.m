% gtversion 1.0
% guitar instrument definition

SR = 48000;

material_tab = [7850 2e11;19300 7.9e10; 19050 2.08e11]; % 1: steel, 2: gold, 3: uranium

notes = [-8 -3 2 7 11 16]';                             % fundamental, in semitones � from middle C
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

ssconnect_def = [0.003784982188670, 10571.66948242946, 12.922073295595544, 2.357470309715547, 0.00000706046088019609, 1, 0.694828622975817, 0, 0.765516788149002; 0.006468815192050, 5853.75648722841, 14.594924263929030, 2.515480261156667, 0.00000031832846377421, 2, 0.317099480060861, 1, 0.795199901137063; 0.010575068354343, 9002.80468888800, 11.557406991565868, 2.486264936249833, 0.00000276922984960890, 3, 0.950222048838355, 2, 0.186872604554379; 0.010648885351993, 2418.86338627215, 5.357116785741896, 1.784454039068336, 0.00000046171390631154, 4, 0.034446080502909, 3, 0.489764395788231; 0.002576130816775, 5217.61282626275, 13.491293058687772, 2.310955780355113, 0.00000097131781235848, 5, 0.438744359656398, 4, 0.445586200710899; 0.010705927817606, 10157.35525189067, 14.339932477575505, 1.342373375623124, 0.00000823457828327293, 6, 0.381558457093008, 5, 0.646313010111265];
