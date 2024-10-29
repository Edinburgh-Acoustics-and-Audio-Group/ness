% SCORE FILE.
% Define the bow gestures here. Set the breakpoint times and values for bow position, force, velocity.

% Charlotte Desvages, 2015.

Tf = 10;                % simulation duration (seconds)

% bowgest is a structure containing each bow's data.
bowgest = struct;
ttb = 0.05;	% transition time for bow gestures

% Bow 1
	bowgest(1).stringnumber	= 1;
	bowgest(1).w0						= 1e-8;
	bowgest(1).vw0					= 0;
	bowgest(1).u0						= 0;
	bowgest(1).vu0					= 0;
	bowgest(1).times				= [0		ttb		0.5		0.5+ttb	Tf		];
	bowgest(1).pos					= [0.86	0.86	0.86	0.86		0.9	];
	bowgest(1).force_w			= [0		-0.3	-0.3	-0.3	-0.4	];
	bowgest(1).force_u			= [0		0			0			2			2.5	];

% Bow 2
	bowgest(2).stringnumber	= 2;
	bowgest(2).w0						= 1e-8;
	bowgest(2).vw0					= 0;
	bowgest(2).u0						= 0;
	bowgest(2).vu0					= 0;
	bowgest(2).times				= [0		ttb		0.5		0.5+ttb	Tf		];
	bowgest(2).pos					= [0.86	0.86	0.86	0.86		0.89	];
	bowgest(2).force_w			= [0		-0.3	-0.3	-0.3	-0.41	];
	bowgest(2).force_u			= [0		0			0			2			2.43	];

%% Bow 2
	%bowgest(2).stringnumber	= 2;
	%bowgest(2).w0						= 1e-8;
	%bowgest(2).v0						= 0;
	%bowgest(2).times				= [0		ttb		0.5		0.5+ttb	Tf		];
	%bowgest(2).pos					= [0.86	0.86	0.85	0.845		0.65	];
	%bowgest(2).force				= [0		-0.3	-0.32	-0.325	-0.2	];
	%bowgest(2).vel					= [0		0			0			0.2			0.15	];

%% Bow 3
	%bowgest(3).stringnumber	= 3;
	%bowgest(3).w0						= 1e-8;
	%bowgest(3).v0						= 0;
	%bowgest(3).times 				= [0 ttb 0.5 0.5+ttb 1 1.5 2.5 2.5+ttb Tf];
	%bowgest(3).pos 					= [0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86 0.86];
	%bowgest(3).force 				= [-0.0 -0.3 -0.3 -0.3 -0.3 -0.3 -0.3 5 5];
	%%bowgest(3).vel			 		= [0 0 0 1 1 1 1 1 1];
	%bowgest(3).vel 					= [0 0 0 0 0 0 0 0 0];

finggest = struct;
ttf = 0.1;

		finggest(1).stringnumber	= 1;
		finggest(1).w0						= 1e-8;
		finggest(1).vw0						= 0;
		finggest(1).u0						= 0;
		finggest(1).vu0						= 0;
		finggest(1).times					= [0			ttb			Tf];
		finggest(1).pos						= [0.159103584746285 0.159103584746285 0.159103584746285];
		finggest(1).force_w				= [0			-3			-3];
		finggest(1).force_u				= [0			0				0];
	
	% Finger 2
		finggest(2).stringnumber	= 2;
		finggest(2).w0						= 1e-8;
		finggest(2).vw0						= 0;
		finggest(2).u0						= 0;
		finggest(2).vu0						= 0;
		finggest(2).times					= [0			ttb			Tf];
		finggest(2).pos						= [0.250846461561659 0.250846461561659 0.579551792373143];
		finggest(2).force_w				= [0			-3			-3];
		finggest(2).force_u				= [0			0				0];
	
	% define vibrato gesture:
		% vibrato = {[begintime	endtime	ramptime	amplitude	frequency]; repeat for several vibrati}.
		%finggest(1).vibrato				= {};
		finggest(1).vibrato				= [0.8	2	0.5	0.02	5];
		finggest(2).vibrato				= [];
