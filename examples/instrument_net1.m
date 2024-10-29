% guitar instrument definition

SR=48000;

% define strings

string_def = [0.68 1 12.1 0.0002 15 5; 0.68 1 12.3 0.00015 15 5; 0.68 1 21.9 0.00015 15 5; 0.68 1 39.2 0.00015 15 7; 0.68 1 27.6 0.0001 15 5; 0.68 1 49.2 0.0001 15 8];

% parameters per row are:
% length (m)
% material type
% tension (N)
% radius (m)
% T60 at DC (s)
% T60 at 1 kHz

% define outputs

output_def = [1 0.9; 2 0.9; 3 0.9; 4 0.9; 5 0.9; 6 0.9];

% parameters per row are
% string number
% position along string (0-1)

% number of iterations for NR: can be played with! Generally need above
% about 15

itnum = 20;
normalize_outs = 1;             % individually normalise output channels

% panning (-1 to 1, per channel)
pan = [0.314723686393179, 0.405791937075619, -0.373013183706494, 0.413375856139019, 0.132359246225410, -0.402459595000590];

%connection network
ssconnect_def = [0.003784982188670, 10571.66948242946, 12.922073295595544, 2.357470309715547, 0.00000706046088019609, 1, 0.694828622975817, 0, 0.765516788149002; 0.006468815192050, 5853.75648722841, 14.594924263929030, 2.515480261156667, 0.00000031832846377421, 2, 0.317099480060861, 1, 0.795199901137063; 0.010575068354343, 9002.80468888800, 11.557406991565868, 2.486264936249833, 0.00000276922984960890, 3, 0.950222048838355, 2, 0.186872604554379; 0.010648885351993, 2418.86338627215, 5.357116785741896, 1.784454039068336, 0.00000046171390631154, 4, 0.034446080502909, 3, 0.489764395788231; 0.002576130816775, 5217.61282626275, 13.491293058687772, 2.310955780355113, 0.00000097131781235848, 5, 0.438744359656398, 4, 0.445586200710899; 0.010705927817606, 10157.35525189067, 14.339932477575505, 1.342373375623124, 0.00000823457828327293, 6, 0.381558457093008, 5, 0.646313010111265];

% parameters per row are
% mass (kg), >0. Do not set to 0!
% frequency (rad), >0. try to keep below about 1e4
% loss parameter (bigger means more loss). >=0
% collision exponent (>1, usually <3). Probably best not to use 1 exactly
% rattling distance (m), >=0. Can be zero!
% string index 1 
% connection point 1 (0-1)
% string index 2
% connection point 2

%Notes:

%if one of the string indeces is 0, then the connection is to one
%string only

%both string indeces can be the same!

% if collision exponent is 1, and rattling distance is 0, the connection is
% linear! Can use this for reverb modeling. 

% I think the code works if two connection points for distinct connections
% are the same...but better to be safe and disallow this!
