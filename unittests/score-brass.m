global T pressure lip_frequency Sr mu sigma H w FS vibamp vibfreq tremamp ...
    tremfreq noiseamp valveopening valvevibfreq valvevibamp instrumentfile maxout

maxout=0.95;
instrumentfile='Trombonevalve';
FS=44100;%Sample rate Hz
T=1;%Length of score s
%Lip area, mass, damping, equilibrium separation, width
Sr=[0,1.46e-5];
mu=[0,5.37e-5];
sigma=[0,5];
H=[0,0.00029];
w=[0,0.01];
%mouth pressure pressure as time(s) pressure(Pa) pairs [t,p]
%assumes that if final time entry is less than length of score pressure
%remains the same as final pressure entry
 pressure=[0,0;
    10e-3,3e3];
%lip frequency as time(s) frequency(Hz) pairs [t,f]
lip_frequency=[0,35];
%vibrato amplitude and frequency
vibamp=[0,0.01];%fraction of normal frequency
vibfreq=[0,1000];%frequency of vibrato
tremamp=[0,0];%fraction of normal mouth pressure
tremfreq=[0,44099];%frequecny of tremolo
noiseamp=[0,0];%fraction of normal mouth pressure
valvevibfreq=[0,100];%valve vibrato frequency
valvevibamp=[0,0];%valve vibrato amplitude
%default tube openings(0-1) 1st column time, 2nd-1st valve, 3rd-2nd valve etc
valveopening=[0,1];
