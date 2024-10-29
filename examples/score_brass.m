%This creates a phrase involving valve transitions. Articulation is added 
%by changing the mouth pressure and modifying the vibrato through time

%usable instruments: horn3, horn3b, trombone3, trombone3b, trumpet3,
%trumpet3b, trumpetbulge3, trumpetbulge3b

maxout=0.95;%max normalised value of instrument output
T=10;%Length of score s, if this is less than a time parameter in score values it is changed to longest time value

%score parameters take time value pairs, if final time entry is less than
%length of score it is assumed to remain constant from that time
%Lip area, mass, damping, equilibrium separation, width
Sr=[0,1.46e-5];
mu=[0,5.37e-5];
sigma=[0,10];
H=[0,0.00029];
w=[0,0.01];
%lip frequency as time(s) frequency(Hz) pairs [t,f]
lip_frequency=[0,700;
    0.49,700;
    0.5,550;
    0.99,550;
    1,700;
    4.3,700;
    4.31,570;%gliss up to note
    4.6,620;
    4.9,620;
    5,550;
    5.39,550;
    5.4,620;
    5.99,620;
    6,700;
    6.39,700;
    6.4,550;
    8.29,550;
    8.3,470];
%mouth pressure pressure as time(s) pressure(Pa) pairs [t,p]
%useful to have small pressure ramp to get better attack
pressure=[0,0;
    1e-3,3e3;
    0.97,3e3;
    1,0;
    1.001,3e3;
    2.7,4e3;%swell
    4,1e3;
    4.1,0;%small rest
    4.3,0;
    4.301,3e3;
    4.9,3e3;
    5,0;
    5.001,3e3;
    5.3,3e3;
    5.4,0;
    5.401,3e3;
    5.9,3e3;
    6,0;
    6.001,3e3;
    6.3,3e3;
    6.4,0;
    6.401,3e3;
    6.9,3e3;
    7,0;
    7.001,3e3;
    7.9,3e3;
    8.3,0;
    8.301,2e3;
    8.99,1.8e3;%diminuendo
    10,0];
%vibrato amplitude and frequency
vibamp=[0,0.01;%fraction of normal frequency
    1.47,0.01;%small amount of vibrato on notes, adds natural element, should be adjusted though
    1.5,0.02;
    3.9,0.02;
    4.2,0.01;
    6,0.02;
    8,0.01];
vibfreq=[0,5;%frequency of vibrato
    2.51,3.3;
    2.53,4;%slow down vibrato
    3.9,3.3;
    8.3,3.3;
    8.5,5];
tremamp=[0,0];%fraction of normal mouth pressure
tremfreq=[0,0];%frequecny of tremolo
noiseamp=[0,0];%fraction of normal mouth pressure
%default tube openings(0-1) 1st column time, 2nd-1st valve, 3rd-2nd valve etc
valveopening=[0,1,1,1;%T,ST,T+ST
    0.45,1,1,1;
    0.5,0,1,0;
    0.97,0,1,0;
    1,1,1,1;
    4.3,1,1,1;
    4.33,0,1,1;
    4.95,0,1,1;
    5,1,0,0;
    5.35,1,0,0;
    5.4,0,1,1;
    5.97,0,1,1;
    6,1,1,1;
    6.35,1,1,1;
    6.4,1,0,0;
    8.29,1,0,0;
    8.3,0,0,0];
valvevibfreq=[0,0,0,0];%valve vibrato frequency
valvevibamp=[0,0,0,0];%valve vibrato amplitude