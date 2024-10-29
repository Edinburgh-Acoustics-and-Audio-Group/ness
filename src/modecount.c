/*
 * Program to count how many modes are required for a modal plate of certain
 * properties, in order to run at a given sampling rate.
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* comparison function for qsort */
static int omega_compare(const void *z1, const void *z2)
{
    double *zazi1 = (double *)z1;
    double *zazi2 = (double *)z2;
    if (zazi1[0] < zazi2[0]) return -1;
    if (zazi1[0] > zazi2[0]) return 1;

    if (zazi1[1] < zazi2[1]) return -1;
    if (zazi1[1] > zazi2[1]) return 1;

    if (zazi1[2] < zazi2[2]) return -1;
    if (zazi1[2] > zazi2[2]) return 1;
    return 0;
}

int main(int argc, char *argv[])
{
    double Lx, Ly, h, Young, nu, rho, fs;

    double *zazi;
    double gf, D;
    int ind, m, n, DD;

    int fsi;
    double fs_lim, fsd;

    if (argc != 8) {
	fprintf(stderr, "Usage: %s <Lx> <Ly> <thickness> <Young's modulus> <Poisson's ratio> <density> <sample rate>\n", argv[0]);
	return 1;
    }
    Lx = atof(argv[1]);
    Ly = atof(argv[2]);
    h = atof(argv[3]);
    Young = atof(argv[4]);
    nu = atof(argv[5]);
    rho = atof(argv[6]);
    fs = atof(argv[7]);

    /* first compute whole omega array */
    D = Young * (h*h*h) / 12.0 / (1.0 - (nu*nu));
    DD = 200;

    zazi = malloc(DD * DD * 3 * sizeof(double));

    ind = 0;
    for (m = 1; m <= DD; m++) {
	for (n = 1; n <= DD; n++) {
	    gf = sqrt(D/rho/h) * ((((double)m)*M_PI/Lx)*(((double)m)*M_PI/Lx) +
				  (((double)n)*M_PI/Ly)*(((double)n)*M_PI/Ly));
	    zazi[(ind*3)+0] = gf;
	    zazi[(ind*3)+1] = (double)m;
	    zazi[(ind*3)+2] = (double)n;
	    ind++;
	}
    }

    /* sort modes into order */
    qsort(zazi, DD*DD, 3*sizeof(double), omega_compare);

    /* now scan through and work out how many modes we need */
    for (m = 0; m < (DD*DD); m++) {
	fs_lim = zazi[(m*3)+0] / 2.0;
	fsi = (int)((2.0 * fs_lim) + 0.5);
	fsd = (double)fsi;

	if (fsd >= fs) break;
    }
    if (m < (DD*DD)) {
	printf("%d modes required, fs=%d\n", (m+1), fsi);
    }
    else {
	printf("%d modes insufficient, fs=%d\n", m, fsi);
    }

    free(zazi);

    return 0;
}
