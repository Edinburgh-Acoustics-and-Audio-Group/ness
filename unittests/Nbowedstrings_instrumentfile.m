% INSTRUMENT FILE.
% Set here the instrument parameters. You can set them all manually, or alternatively, select an instrument from the database (violin, viola, cello).

% Returns:
% - Fs, audio sample rate ; Tf, simulation duration;
% - "strings", a structure array containing all the strings parameters;
% - "bow" containing the bow parameters

% Charlotte Desvages, 2015

%%%%%%%% AUDIO PARAMETERS

	Fs = 44100;		% audio sample rate (Hz)

%%%%%%%% STRING PARAMETERS

        flag.manual = 0;

	% string_data.mat contains data for 20 different violin strings (5 at each pitch), 8 viola strings (2 at each pitch) and 12 cello strings (3 at each pitch), extracted from Percival's PhD thesis (2013).
	Nstrings = 4;					% number of strings

	instrument = 'viola';	% select instrument type (violin, viola, cello)
	instr_numb = 1;					% select the instrument (violin: 1-5, viola: 1-2, cello: 1-3)

	% Set parameters manually:
	%case 1
		% First, set the number of strings
	%	Nstrings = 1;

		% String parameters:
		% f0 (Hz): fundamental frequency
		% rho (kg/m): linear mass
		% rad (m): radius
		% E (Pa): Young's modulus
		% T60(DC) (s): T60 at DC
		% T60(1kHz) (s): T60 at 1kHz
		% L (m): length
	%	strings = struct;

		% String 1
	%		strings(1).f0 = 440;
	%		strings(1).rho = 7e-4;
	%		strings(1).rad = 3e-4;
	%		strings(1).E = 2e11;
	%		strings(1).T60 = [10;8];
	%		strings(1).L = 0.35;

		%% String 2
			%strings(2).f0 = 180;
			%strings(2).rho = 8e-4;
			%strings(2).rad = 1e-4;
			%strings(2).E = 2e11;
			%strings(2).T60 = [10;8];
			%strings(2).L = 0.34;

		%% String 3
			%strings(3).f0 = 220;
			%strings(3).rho = 8e-4;
			%strings(3).rad = 2e-3;
			%strings(3).E = 2e11;
			%strings(3).T60 = [10;8];
			%strings(3).L = 0.34;

%%%%%%%% BOW PARAMETERS

	bow = struct();
	bow.Kw			= 1e6;
	bow.alpha		= 2.0;
	bow.beta		= 20;
	bow.lambda 	= 10;
	bow.M				= 0.1;

	fing = struct();
	fing.Kw = 1e5;
	fing.Ku			= 1e3;
	fing.alpha	= 2.2;
	fing.beta		= 50;
	fing.lambda	= 20;
	fing.M			= 0.05;
