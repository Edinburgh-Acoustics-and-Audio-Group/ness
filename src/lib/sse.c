#include "banded.h"
#include "pcg.h"

#include <xmmintrin.h>
#include <emmintrin.h>

void m5x5_vector_SSE(matrix_5x5_t *m5x5, double *in, double *out)
{
    int i;
    int N = m5x5->N;
    int h = m5x5->h;
    int hh = h + h;
    double *val = m5x5->values;

    __m128d in0, in1, in2, in3, in4, in5, in6, in7, in8, in9;
    __m128d total, stencil, scratch, inx;

    /* row of 13 */
    *out = in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 14 */
    *out = in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* loop over rows of 15 */
    for (i = 0; i < (h-4); i++) {
	*out = in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	    + in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	    + in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
	out++;
	in++;
	val += 25;
    }

    /* row of 16 */
    *out = in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 17 */
    *out = in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 18 */
    *out = in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 19 */
    *out = in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* loop over rows of 20 */
    for (i = 0; i < (h-4); i++) {
	*out = in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	    + in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	    + in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	    + in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
	out++;
	in++;
	val += 25;
    }

    /* row of 21 */
    *out = in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 22 */
    *out = in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 23 */
    *out = in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* row of 24 */
    *out = in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
    out++;
    in++;
    val += 25;

    /* main loop over middle rows in pairs */
    i = (N - (h << 2) - 4) >> 1;

    in0 = _mm_loadu_pd(&in[-hh-1]);
    in2 = _mm_loadu_pd(&in[-h-1]);
    in4 = _mm_loadu_pd(&in[-1]);
    in6 = _mm_loadu_pd(&in[h-1]);
    in8 = _mm_loadu_pd(&in[hh-1]);

    while (i > 0) {
	/* row 1, -2H */
	inx = _mm_load_sd(&in[-hh-2]);
	scratch = _mm_load_sd(val);
	scratch = _mm_mul_sd(scratch, inx);

	total = _mm_loadu_pd(val + 1);
	total = _mm_mul_pd(total, in0);
	total = _mm_add_sd(total, scratch);

	in1 = _mm_loadu_pd(&in[-hh+1]);
	stencil = _mm_loadu_pd(val + 3);
	stencil = _mm_mul_pd(stencil, in1);
	total = _mm_add_pd(total, stencil);

	/* row 1, -H */
	inx = _mm_load_sd(&in[-h-2]);
	scratch = _mm_load_sd(val + 5);
	scratch = _mm_mul_sd(scratch, inx);
	total = _mm_add_sd(total, scratch);

	stencil = _mm_loadu_pd(val + 6);
	stencil = _mm_mul_pd(stencil, in2);
	total = _mm_add_pd(total, stencil);

	in3 = _mm_loadu_pd(&in[-h+1]);
	stencil = _mm_loadu_pd(val + 8);
	stencil = _mm_mul_pd(stencil, in3);
	total = _mm_add_pd(total, stencil);
	
	/* row 1, 0 */
	inx = _mm_load_sd(&in[-2]);
	scratch = _mm_load_sd(val + 10);
	scratch = _mm_mul_sd(scratch, inx);
	total = _mm_add_sd(total, scratch);

	stencil = _mm_loadu_pd(val + 11);
	stencil = _mm_mul_pd(stencil, in4);
	total = _mm_add_pd(total, stencil);

	in5 = _mm_loadu_pd(&in[1]);
	stencil = _mm_loadu_pd(val + 13);
	stencil = _mm_mul_pd(stencil, in5);
	total = _mm_add_pd(total, stencil);
	
	/* row 1, +H */
	inx = _mm_load_sd(&in[h-2]);
	scratch = _mm_load_sd(val + 15);
	scratch = _mm_mul_sd(scratch, inx);
	total = _mm_add_sd(total, scratch);

	stencil = _mm_loadu_pd(val + 16);
	stencil = _mm_mul_pd(stencil, in6);
	total = _mm_add_pd(total, stencil);

	in7 = _mm_loadu_pd(&in[h+1]);
	stencil = _mm_loadu_pd(val + 18);
	stencil = _mm_mul_pd(stencil, in7);
	total = _mm_add_pd(total, stencil);
	
	/* row 1, +2H */
	inx = _mm_load_sd(&in[hh-2]);
	scratch = _mm_load_sd(val + 20);
	scratch = _mm_mul_sd(scratch, inx);
	total = _mm_add_sd(total, scratch);

	stencil = _mm_loadu_pd(val + 21);
	stencil = _mm_mul_pd(stencil, in8);
	total = _mm_add_pd(total, stencil);

	in9 = _mm_loadu_pd(&in[hh+1]);
	stencil = _mm_loadu_pd(val + 23);
	stencil = _mm_mul_pd(stencil, in9);
	total = _mm_add_pd(total, stencil);
	
	scratch = _mm_shuffle_pd(total, total, 3);
	total = _mm_add_sd(total, scratch);
	_mm_store_sd(out, total);

	out++;
	in++;
	val += 25;

	/* row 2, -2H */
	total = _mm_loadu_pd(val);
	total = _mm_mul_pd(total, in0);

	stencil = _mm_loadu_pd(val + 2);
	stencil = _mm_mul_pd(stencil, in1);
	total = _mm_add_pd(total, stencil);

	inx = _mm_load_sd(&in[-hh+2]);
	stencil = _mm_load_sd(val + 4);
	stencil = _mm_mul_sd(stencil, inx);
	total = _mm_add_sd(total, stencil);

	/* row 2, -H */
	stencil = _mm_loadu_pd(val + 5);
	stencil = _mm_mul_pd(stencil, in2);
	total = _mm_add_pd(total, stencil);

	stencil = _mm_loadu_pd(val + 7);
	stencil = _mm_mul_pd(stencil, in3);
	total = _mm_add_pd(total, stencil);

	inx = _mm_load_sd(&in[-h+2]);
	stencil = _mm_load_sd(val + 9);
	stencil = _mm_mul_sd(stencil, inx);
	total = _mm_add_sd(total, stencil);

	/* row 2, 0 */
	stencil = _mm_loadu_pd(val + 10);
	stencil = _mm_mul_pd(stencil, in4);
	total = _mm_add_pd(total, stencil);

	stencil = _mm_loadu_pd(val + 12);
	stencil = _mm_mul_pd(stencil, in5);
	total = _mm_add_pd(total, stencil);

	inx = _mm_load_sd(&in[2]);
	stencil = _mm_load_sd(val + 14);
	stencil = _mm_mul_sd(stencil, inx);
	total = _mm_add_sd(total, stencil);

	/* row 2, H */
	stencil = _mm_loadu_pd(val + 15);
	stencil = _mm_mul_pd(stencil, in6);
	total = _mm_add_pd(total, stencil);

	stencil = _mm_loadu_pd(val + 17);
	stencil = _mm_mul_pd(stencil, in7);
	total = _mm_add_pd(total, stencil);

	inx = _mm_load_sd(&in[h+2]);
	stencil = _mm_load_sd(val + 19);
	stencil = _mm_mul_sd(stencil, inx);
	total = _mm_add_sd(total, stencil);

	/* row 2, 2H */
	stencil = _mm_loadu_pd(val + 20);
	stencil = _mm_mul_pd(stencil, in8);
	total = _mm_add_pd(total, stencil);

	stencil = _mm_loadu_pd(val + 22);
	stencil = _mm_mul_pd(stencil, in9);
	total = _mm_add_pd(total, stencil);

	inx = _mm_load_sd(&in[hh+2]);
	stencil = _mm_load_sd(val + 24);
	stencil = _mm_mul_sd(stencil, inx);
	total = _mm_add_sd(total, stencil);

	scratch = _mm_shuffle_pd(total, total, 3);
	total = _mm_add_sd(total, scratch);
	_mm_store_sd(out, total);

	out++;
	in++;
	val += 25;

	in0 = in1;
	in2 = in3;
	in4 = in5;
	in6 = in7;
	in8 = in9;

	i--;
    }
    if (N & 1) {
	*out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	    + in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	    + in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	    + in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	    + in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23] + in[hh+2]*val[24];
	out++;
	in++;
	val += 25;
    }

    /* row of 24 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22] + in[hh+1]*val[23];
    out++;
    in++;
    val += 25;

    /* row of 23 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21] + in[hh]*val[22];
    out++;
    in++;
    val += 25;

    /* row of 22 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20] + in[hh-1]*val[21];
    out++;
    in++;
    val += 25;

    /* row of 21 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19]
	+ in[hh-2]*val[20];
    out++;
    in++;
    val += 25;

    /* loop over rows of 20 */
    for (i = 0; i < (h-4); i++) {
	*out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	    + in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	    + in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	    + in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18] + in[h+2]*val[19];
	out++;
	in++;
	val += 25;
    }

    /* row of 19 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17] + in[h+1]*val[18];
    out++;
    in++;
    val += 25;

    /* row of 18 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16] + in[h]*val[17];
    out++;
    in++;
    val += 25;

    /* row of 17 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15] + in[h-1]*val[16];
    out++;
    in++;
    val += 25;

    /* row of 16 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14]
	+ in[h-2]*val[15];
    out++;
    in++;
    val += 25;

    /* loop over rows of 15 */
    for (i = 0; i < (h-4); i++) {
	*out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	    + in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	    + in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13] + in[2]*val[14];
	out++;
	in++;
	val += 25;
    }

    /* row of 14 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12] + in[1]*val[13];
    out++;
    in++;
    val += 25;

    /* row of 13 */
    *out = in[-hh-2]*val[0] + in[-hh-1]*val[1] + in[-hh]*val[2] + in[-hh+1]*val[3] + in[-hh+2]*val[4]
	+ in[-h-2]*val[5] + in[-h-1]*val[6] + in[-h]*val[7] + in[-h+1]*val[8] + in[-h+2]*val[9]
	+ in[-2]*val[10] + in[-1]*val[11] + in[0]*val[12];
    out++;
    in++;
    val += 25;
}

void m3x3_vector_SSE(matrix_3x3_t *m3x3, double *in, double *out)
{
    int i;
    int N = m3x3->N;
    int h = m3x3->h;
    double *val = m3x3->values;

    __m128d in0, in1, in2, in3, in4, in5;
    __m128d stencil0, stencil1, stencil2, stencil3, stencil4, stencil5;
    __m128d total, scratch;

    /* first row - 5 values */
    out[0] = in[0]*val[4] + in[1]*val[5] + in[h-1]*val[6]
	+ in[h]*val[7] + in[h+1]*val[8];
    out++;
    in++;
    val += 9;

    /* H-2 rows of 6 values */
    for (i = 0; i < (h-2); i++) {
	in0 = _mm_loadu_pd(&in[-1]);
	stencil0 = _mm_loadu_pd(val + 3);
	total = _mm_mul_pd(stencil0, in0);
	
	in1 = _mm_load_sd(&in[1]);
	in1 = _mm_loadh_pd(in1, &in[h-1]);
	stencil1 = _mm_loadu_pd(val + 5);
	stencil1 = _mm_mul_pd(stencil1, in1);
	total = _mm_add_pd(total, stencil1);

	in2 = _mm_loadu_pd(&in[h]);
	stencil2 = _mm_loadu_pd(val + 7);
	stencil2 = _mm_mul_pd(stencil2, in2);
	total = _mm_add_pd(total, stencil2);

	scratch = _mm_shuffle_pd(total, total, 3);
	total = _mm_add_sd(total, scratch);
	_mm_store_sd(out, total);

	out++;
	in++;
	val += 9;
    }

    /* row of 7 values */
    *out = in[-h+1]*val[2] + in[-1]*val[3] + in[0]*val[4] + in[1]*val[5]
	+ in[h-1]*val[6] + in[h]*val[7] + in[h+1]*val[8];
    out++;
    in++;
    val += 9;

    /* row of 8 values */
    *out = in[-h]*val[1] + in[-h+1]*val[2] + in[-1]*val[3] + in[0]*val[4]
	+ in[1]*val[5] + in[h-1]*val[6] + in[h]*val[7] + in[h+1]*val[8];
    out++;
    in++;
    val += 9;

    /* main double loop over rows with all 9 values */
    i = (N >> 1) - h - 1;
    while (i > 0) {
	in0 = _mm_loadu_pd(&in[-h-1]);
	stencil0 = _mm_loadu_pd(val);
	total = _mm_mul_pd(stencil0, in0);

	in1 = _mm_loadu_pd(&in[-h+1]);
	stencil1 = _mm_load_sd(val + 2);
	stencil1 = _mm_mul_sd(stencil1, in1);
	total = _mm_add_sd(total, stencil1);

	in2 = _mm_loadu_pd(&in[-1]);
	stencil2 = _mm_loadu_pd(val + 3);
	stencil2 = _mm_mul_pd(stencil2, in2);
	total = _mm_add_pd(total, stencil2);

	in3 = _mm_loadu_pd(&in[1]);
	stencil3 = _mm_load_sd(val + 5);
	stencil3 = _mm_mul_sd(stencil3, in3);
	total = _mm_add_sd(total, stencil3);

	in4 = _mm_loadu_pd(&in[h-1]);
	stencil4 = _mm_loadu_pd(val + 6);
	stencil4 = _mm_mul_pd(stencil4, in4);
	total = _mm_add_pd(total, stencil4);

	in5 = _mm_loadu_pd(&in[h+1]);
	stencil5 = _mm_load_sd(val + 8);
	stencil5 = _mm_mul_sd(stencil5, in5);
	total = _mm_add_sd(total, stencil5);

	scratch = _mm_shuffle_pd(total, total, 3);
	total = _mm_add_sd(total, scratch);
	_mm_store_sd(out, total);

	/* second row */
	stencil0 = _mm_load_sd(val + 9);
	in0 = _mm_shuffle_pd(in0, in0, 3);
	stencil0 = _mm_mul_sd(stencil0, in0);

	stencil1 = _mm_loadu_pd(val + 10);
	total = _mm_mul_pd(stencil1, in1);
	total = _mm_add_sd(total, stencil0);

	stencil2 = _mm_load_sd(val + 12);
	in2 = _mm_shuffle_pd(in2, in2, 3);
	stencil2 = _mm_mul_pd(stencil2, in2);
	total = _mm_add_pd(total, stencil2);

	stencil3 = _mm_loadu_pd(val + 13);
	stencil3 = _mm_mul_pd(stencil3, in3);
	total = _mm_add_pd(total, stencil3);

	stencil4 = _mm_load_sd(val + 15);
	in4 = _mm_shuffle_pd(in4, in4, 3);
	stencil4 = _mm_mul_pd(stencil4, in4);
	total = _mm_add_pd(total, stencil4);

	stencil5 = _mm_loadu_pd(val + 16);
	stencil5 = _mm_mul_pd(stencil5, in5);
	total = _mm_add_pd(total, stencil5);

	scratch = _mm_shuffle_pd(total, total, 3);
	total = _mm_add_sd(total, scratch);
	_mm_store_sd(out + 1, total);

	out += 2;
	in += 2;
	val += 18;
	
	i--;
    }
    if (N & 1) {
	/* odd row */
	*out = in[-h-1]*val[0] + in[-h]*val[1] + in[-h+1]*val[2]
	    + in[-1]*val[3] + in[0]*val[4] + in[1]*val[5]
	    + in[h-1]*val[6] + in[h]*val[7] + in[h+1]*val[8];
	out++;
	in++;
	val += 9;
    }
	
    /* 8 values row */
    *out = in[-h-1]*val[0] + in[-h]*val[1] + in[-h+1]*val[2]
	+ in[-1]*val[3] + in[0]*val[4] + in[1]*val[5]
	+ in[h-1]*val[6] + in[h]*val[7];
    out++;
    in++;
    val += 9;

    /* 7 values row */
    *out = in[-h-1]*val[0] + in[-h]*val[1] + in[-h+1]*val[2]
	+ in[-1]*val[3] + in[0]*val[4] + in[1]*val[5]
	+ in[h-1]*val[6];
    out++;
    in++;
    val += 9;
    
    /* loop over rows with 6 values */
    for (i = 0; i < (h-2); i++) {
	*out = in[-h-1]*val[0] + in[-h]*val[1] + in[-h+1]*val[2]
	    + in[-1]*val[3] + in[0]*val[4] + in[1]*val[5];
	out++;
	in++;
	val += 9;
    }

    /* 5 values row */
    *out = in[-h-1]*val[0] + in[-h]*val[1] + in[-h+1]*val[2]
	+ in[-1]*val[3] + in[0]*val[4];
}

void backwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x)
{
    int i, j;
    int N = dec->L->nrow;
    int bs = dec->bandSize;
    int start = 4;

    float *ld = dec->Udense;
    float *diag = &dec->diag[N - 8];
    float *xs;
    float val;

    __m128 total, xsrc, ldense, scratch, di;
    __m128 x0, x1, x2;

    /* point to offset for first loop iteration */
    rhs = &rhs[N - 8];

    /* do bottom 4 rows first */
    x[N-1] = rhs[7] * diag[7];
    x[N-2] = (rhs[6] - (ld[0] * x[N-1])) * diag[6];
    x[N-3] = (rhs[5] - (ld[1] * x[N-1]) - (ld[2] * x[N-2])) * diag[5];
    x[N-4] = (rhs[4] - (ld[3] * x[N-1]) - (ld[4] * x[N-2]) - (ld[5] * x[N-3])) * diag[4];

    ld += 8;

    /* loop over rows in blocks of 4 */
    for (i = N-8; i >= 0; i -= 4) {
	total = _mm_loadu_ps(rhs);
	rhs -= 4;

	if (start < bs) {
	    j = (start >> 2);
	    xs = &x[N - 4];
	    start += 4;
	}
	else {
	    xs = &x[(i + bs + 4) - 4];
	    j = (bs >> 2);
	}

	while (j > 0) {
	    xsrc = _mm_loadu_ps(xs);
	    xs -= 4;

	    ldense = _mm_loadu_ps(ld);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0xff);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 4);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0xaa);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 8);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0x55);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 12);
	    ld += 16;
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0x00);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);
	    
	    j--;
	}

	/* triangle bit on LHS */
	di = _mm_load_ss(diag+3);
	x0 = _mm_shuffle_ps(total, total, 0xff);
	x0 = _mm_mul_ss(x0, di);
	_mm_store_ss(&x[i+3], x0);    /* store first x */

	scratch = total;
	scratch = _mm_shuffle_ps(scratch, scratch, 0xaa);
	ldense = _mm_load_ss(ld);
	ldense = _mm_mul_ss(ldense, x0);
	di = _mm_load_ss(diag+2);
	scratch = _mm_sub_ss(scratch, ldense);
	x1 = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i+2], x1);

	scratch = total;
	scratch = _mm_shuffle_ps(scratch, scratch, 0x55);
	ldense = _mm_load_ss(ld+1);
	ldense = _mm_mul_ss(ldense, x0);
	scratch = _mm_sub_ss(scratch, ldense);
	ldense = _mm_load_ss(ld+2);
	ldense = _mm_mul_ss(ldense, x1);
	scratch = _mm_sub_ss(scratch, ldense);
	di = _mm_load_ss(diag+1);
	x2 = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i+1], x2);

	ldense = _mm_load_ss(ld+3);
	ldense = _mm_mul_ss(ldense, x0);
	scratch = _mm_sub_ss(total, ldense);
	ldense = _mm_load_ss(ld+4);
	ldense = _mm_mul_ss(ldense, x1);
	scratch = _mm_sub_ss(scratch, ldense);
	ldense = _mm_load_ss(ld+5);
	ldense = _mm_mul_ss(ldense, x2);
	scratch = _mm_sub_ss(scratch, ldense);
	di = _mm_load_ss(diag);
	scratch = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i], scratch);

	ld += 8;
	diag -= 4;
    }

    /* do remaining rows at end */
    i &= 3;
    while (i > 0) {
	xs = &x[i + bs - 1];
	j = bs;
	val = rhs[3];
	rhs--;

	while (j > 0) {
	    val -= (*xs--) * (*ld++);
	    j--;
	}
	x[i-1] = val * diag[3];
	diag--;

	i--;
    }
}

void forwardSolveSSESp(decomposition_sp_t *dec, float *rhs, float *x)
{
    int i, j;
    int N = dec->L->nrow;
    int bs = dec->bandSize;
    int start = bs - 4;

    float *ld = dec->Ldense;
    float *diag = dec->diag;
    float *xs;
    float val;

    __m128 total, xsrc, ldense, scratch, di;
    __m128 x0, x1, x2;

    /*
     * First do top 4 rows
     */
    x[0] = rhs[0] * diag[0];
    x[1] = (rhs[1] - (ld[0] * x[0])) * diag[1];
    x[2] = (rhs[2] - (ld[1] * x[0]) - (ld[2] * x[1])) * diag[2];
    x[3] = (rhs[3] - (ld[3] * x[0]) - (ld[4] * x[1]) - (ld[5] * x[2])) * diag[3];

    rhs += 4;
    diag += 4;
    ld += 8;
    
    /* loop over rows in groups of 4 */
    for (i = 4; i < (N-4); i += 4) {
	total = _mm_loadu_ps(rhs);
	rhs += 4;

	if (start) {
	    j = ((bs - start) >> 2);
	    xs = x;
	    start -= 4;
	}
	else {
	    xs = &x[i - bs];
	    j = (bs >> 2);
	}
	
	while (j > 0) {
	    xsrc = _mm_loadu_ps(xs);
	    xs += 4;

	    ldense = _mm_loadu_ps(ld);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 4);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0x55);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 8);
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0xaa);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    ldense = _mm_loadu_ps(ld + 12);
	    ld += 16;
	    scratch = xsrc;
	    scratch = _mm_shuffle_ps(scratch, scratch, 0xff);
	    ldense = _mm_mul_ps(ldense, scratch);
	    total = _mm_sub_ps(total, ldense);

	    j--;
	}

	/* handle triangle bit on RHS */
	di = _mm_load_ss(diag);
	x0 = _mm_mul_ss(total, di);
	_mm_store_ss(&x[i], x0);    /* store first x */

	scratch = total;
	scratch = _mm_shuffle_ps(scratch, scratch, 0x55);
	ldense = _mm_load_ss(ld);
	ldense = _mm_mul_ss(ldense, x0);
	di = _mm_load_ss(diag+1);
	scratch = _mm_sub_ss(scratch, ldense);
	x1 = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i+1], x1);

	scratch = total;
	scratch = _mm_shuffle_ps(scratch, scratch, 0xaa);
	ldense = _mm_load_ss(ld+1);
	ldense = _mm_mul_ss(ldense, x0);
	scratch = _mm_sub_ss(scratch, ldense);
	ldense = _mm_load_ss(ld+2);
	ldense = _mm_mul_ss(ldense, x1);
	scratch = _mm_sub_ss(scratch, ldense);
	di = _mm_load_ss(diag+2);
	x2 = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i+2], x2);

	scratch = total;
	scratch = _mm_shuffle_ps(scratch, scratch, 0xff);
	ldense = _mm_load_ss(ld+3);
	ldense = _mm_mul_ss(ldense, x0);
	scratch = _mm_sub_ss(scratch, ldense);
	ldense = _mm_load_ss(ld+4);
	ldense = _mm_mul_ss(ldense, x1);
	scratch = _mm_sub_ss(scratch, ldense);
	ldense = _mm_load_ss(ld+5);
	ldense = _mm_mul_ss(ldense, x2);
	scratch = _mm_sub_ss(scratch, ldense);
	di = _mm_load_ss(diag+3);
	scratch = _mm_mul_ss(scratch, di);
	_mm_store_ss(&x[i+3], scratch);

	ld += 8;
	diag += 4;
    }

    /* do remaining rows at end */
    while (i < N) {
	xs = &x[i - bs];
	j = bs;
	val = *rhs++;

	while (j > 0) {
	    val -= (*xs++) * (*ld++);
	    j--;
	}
	x[i] = val * (*diag++);

	i++;
    }
}


