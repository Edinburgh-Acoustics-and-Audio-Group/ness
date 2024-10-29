% Sample Matlab format file for parser unit test

x=1.32;

    y  =      -7e-10;

arr1D = [1,2,3,4,5]';

arr2D=[5,6,7;
	8 9 10;
	11,12,13];

cellarray = {
    1, [3 5 7; 9 11 13];
    2, [4 6 8; 10 12 14]
};

frets = fret_def_gen(4, 1, -0.001);

z = (1+sin(pi/2))^3/4;

str1 = struct();
str1.a = arr1D(3);
str1.b = x;
str1.c = 3*3;
str1.d = [ 9,8,7,6,5 ];

str2 = struct;
str2(1).i = 44;
str2(1).j = 55;
str2(2).i = 66;
str2(2).j = 77;

string1 = 'This is a single-quoted string';

w = str2(2).i;
v = cellarray{2,2}(2,1);

arr3 = cellarray{1,2};
