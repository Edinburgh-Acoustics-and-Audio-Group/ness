#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
    FILE *f;
    int len1, len2, i;
    double *dat1, *dat2;
    double tolerance;

    if (argc != 4) {
	fprintf(stderr, "Usage: %s <file 1> <file 2> <tolerance>\n", argv[0]);
	fprintf(stderr, "  Compares two raw double output files to the specified tolerance\n");
	fprintf(stderr, "  Returns 0 if they match, 1 if not\n");
	return 2;
    }

    /* read first file */
    f = fopen(argv[1], "rb");
    if (!f) {
	fprintf(stderr, "Cannot open %s\n", argv[1]);
	return 2;
    }
    fseek(f, 0, SEEK_END);
    len1 = ftell(f);
    fseek(f, 0, SEEK_SET);
    len1 &= 0xfffffff8;

    dat1 = malloc(len1);
    if (!dat1) {
	fprintf(stderr, "Out of memory\n");
	return 2;
    }
    fread(dat1, 1, len1, f);
    fclose(f);

    /* read second file */
    f = fopen(argv[2], "rb");
    if (!f) {
	fprintf(stderr, "Cannot open %s\n", argv[2]);
	return 2;
    }
    fseek(f, 0, SEEK_END);
    len2 = ftell(f);
    fseek(f, 0, SEEK_SET);
    len2 &= 0xfffffff8;

    dat2 = malloc(len2);
    if (!dat2) {
	fprintf(stderr, "Out of memory\n");
	return 2;
    }
    fread(dat2, 1, len2, f);
    fclose(f);

    /* check that the lengths match */
    if (len1 != len2) {
	return 1;
    }

    /* get tolerance */
    tolerance = atof(argv[3]);

    /* compare the data */
    for (i = 0; i < (len1 / 8); i++) {
	if (fabs(dat1[i] - dat2[i]) > tolerance) {
	    return 1;
	}
    }
    return 0;
}
