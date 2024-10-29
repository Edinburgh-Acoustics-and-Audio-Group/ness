%Instrument file for trumpet
global temperature bore vpos vdl vbl custominstrument
custominstrument=0;
temperature=20;%temperature in C
%valve values, assumes that valve is constant cylinder where cross section
%in and out of valves is same as point on instrument
vpos=[600];%position of valve in mm
vdl=[20];%default tube length mm
vbl=[200];%bypass tube length mm (will add default length to either side of bypass tubes)
%bore in axial position (mm) and diameter(mm) pairs [x,d]
%bore=dlmread('SmithWatkinsTrumpet_KellyScrmouthpieceBORE.txt');
bore=[0,25.65;
5,25.65;
10,24.97;
15,22.5;
20,17.5;
25,11.3;
27.5,8;
30,7.25;
44.6,7.7;
49.6,8.1;
54.6,8.4;
59.6,8.9;
64.6,9.4;
69.6,9.9;
74.6,10.65;
79.6,11.1;
84.6,12.8;
89.6,13.6;
689.6,13.6;
764.6,13.6;
919.6,14.8;
1594.6,13.6;
1666.6,13.85;
1694.6,13.85;
1794.6,14.45;
1829.6,14.5;
1931.6,14.9;
2031.6,17.3;
2131.6,19.4;
2138.6,19.42;
2240.6,20;
2328.6,24.5;
2340.6,25.55;
2427.6,30;
2440.6,31.1;
2473.6,33.1;
2513.6,36.6;
2540.6,39.1;
2551.6,40.4;
2583.6,44.7;
2608.1,49.4;
2627.6,54.6;
2640.6,60.6;
2644.1,60.3;
2657.1,66.7;
2668.6,73.8;
2677.6,81.4;
2686.1,90;
2694.6,99.5;
2700.6,109.8;
2707.1,121.3;
2713.1,133.8;
2718.1,148.2;
2723.6,164;
2729.1,181.2;
2733.6,200;
2735.6,214.1;
];
