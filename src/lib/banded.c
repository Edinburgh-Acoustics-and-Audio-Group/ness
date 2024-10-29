/*
 * Fast implementation for banded matrices that implement stencils (9 point and 25 point)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "banded.h"
#include "csrmatrix.h"

matrix_5x5_t *allocate5x5(int N, int h)
{
    matrix_5x5_t *result;
    result = (matrix_5x5_t *)malloc(sizeof(matrix_5x5_t));
    if (!result) return NULL;

    result->N = N;
    result->h = h;

    result->values = (double *)malloc(N * 25 * sizeof(double));
    if (!result->values) {
	free(result);
	return NULL;
    }
    memset(result->values, 0, N * 25 * sizeof(double));

    return result;
}

matrix_5x5_t *m5x5_duplicate(matrix_5x5_t *mat)
{
    matrix_5x5_t *result = allocate5x5(mat->N, mat->h);
    memcpy(result->values, mat->values, mat->N * 25 * sizeof(double));
    return result;
}

void m5x5_vector_mult(matrix_5x5_t *m5x5, double *in, double *out)
{
    int i;
    double val;
    int h = m5x5->h;
    int hh = h+h;
    int N = m5x5->N;
    double *v;

    v = &m5x5->values[0];
    for (i = 0; i < (hh+2); i++) {
	val = 0.0;
	if (i >= (hh+2)) {
	    val += in[i-hh-2] * v[0];
	}
	if (i >= (hh+1)) {
	    val += in[i-hh-1] * v[1];
	}
	if (i >= (hh)) {
	    val += in[i-hh] * v[2];
	}
	if (i >= (hh-1)) {
	    val += in[i-hh+1] * v[3];
	}
	if (i >= (hh-2)) {
	    val += in[i-hh+2] * v[4];
	}
	    
	if (i >= (h+2)) {
	    val += in[i-h-2] * v[5];
	}
	if (i >= (h+1)) {
	    val += in[i-h-1] * v[6];
	}
	if (i >= (h)) {
	    val += in[i-h] * v[7];
	}
	if (i >= (h-1)) {
	    val += in[i-h+1] * v[8];
	}
	if (i >= (h-2)) {
	    val += in[i-h+2] * v[9];
	}
	    
	if (i >= (2)) {
	    val += in[i-2] * v[10];
	}
	if (i >= (1)) {
	    val += in[i-1] * v[11];
	}
	val += in[i] * v[12];
	if (i < (N-1)) {
	    val += in[i+1] * v[13];
	}
	if (i < (N-2)) {
	    val += in[i+2] * v[14];
	}
	    
	if (i < (N-h+2)) {
	    val += in[i+h-2] * v[15];
	}
	if (i < (N-h+1)) {
	    val += in[i+h-1] * v[16];
	}
	if (i < (N-h)) {
	    val += in[i+h] * v[17];
	}
	if (i < (N-h-1)) {
	    val += in[i+h+1] * v[18];
	}
	if (i < (N-h-2)) {
	    val += in[i+h+2] * v[19];
	}
	    
	if (i < (N-hh+2)) {
	    val += in[i+hh-2] * v[20];
	}
	if (i < (N-hh+1)) {
	    val += in[i+hh-1] * v[21];
	}
	if (i < (N-hh)) {
	    val += in[i+hh] * v[22];
	}
	if (i < (N-hh-1)) {
	    val += in[i+hh+1] * v[23];
	}
	if (i < (N-hh-2)) {
	    val += in[i+hh+2] * v[24];
	}

	out[i] = val;
	v += 25;
    }

    for (; i < (N-hh-2); i++) {
	out[i] = in[i-hh-2]*v[0] + in[i-hh-1]*v[1] + in[i-hh]*v[2]
	    + in[i-hh+1]*v[3] + in[i-hh+2]*v[4]
	    + in[i-h-2]*v[5] + in[i-h-1]*v[6] + in[i-h]*v[7]
	    + in[i-h+1]*v[8] + in[i-h+2]*v[9]
	    + in[i-2]*v[10] + in[i-1]*v[11] + in[i]*v[12]
	    + in[i+1]*v[13] + in[i+2]*v[14]
	    + in[i+h-2]*v[15] + in[i+h-1]*v[16] + in[i+h]*v[17]
	    + in[i+h+1]*v[18] + in[i+h+2]*v[19]
	    + in[i+hh-2]*v[20] + in[i+hh-1]*v[21] + in[i+hh]*v[22]
	    + in[i+hh+1]*v[23] + in[i+hh+2]*v[24];
	v += 25;
    }

    for (; i < N; i++) {
	val = 0.0;
	if (i >= (hh+2)) {
	    val += in[i-hh-2] * v[0];
	}
	if (i >= (hh+1)) {
	    val += in[i-hh-1] * v[1];
	}
	if (i >= (hh)) {
	    val += in[i-hh] * v[2];
	}
	if (i >= (hh-1)) {
	    val += in[i-hh+1] * v[3];
	}
	if (i >= (hh-2)) {
	    val += in[i-hh+2] * v[4];
	}
	    
	if (i >= (h+2)) {
	    val += in[i-h-2] * v[5];
	}
	if (i >= (h+1)) {
	    val += in[i-h-1] * v[6];
	}
	if (i >= (h)) {
	    val += in[i-h] * v[7];
	}
	if (i >= (h-1)) {
	    val += in[i-h+1] * v[8];
	}
	if (i >= (h-2)) {
	    val += in[i-h+2] * v[9];
	}
	    
	if (i >= (2)) {
	    val += in[i-2] * v[10];
	}
	if (i >= (1)) {
	    val += in[i-1] * v[11];
	}
	val += in[i] * v[12];
	if (i < (N-1)) {
	    val += in[i+1] * v[13];
	}
	if (i < (N-2)) {
	    val += in[i+2] * v[14];
	}
	    
	if (i < (N-h+2)) {
	    val += in[i+h-2] * v[15];
	}
	if (i < (N-h+1)) {
	    val += in[i+h-1] * v[16];
	}
	if (i < (N-h)) {
	    val += in[i+h] * v[17];
	}
	if (i < (N-h-1)) {
	    val += in[i+h+1] * v[18];
	}
	if (i < (N-h-2)) {
	    val += in[i+h+2] * v[19];
	}
	    
	if (i < (N-hh+2)) {
	    val += in[i+hh-2] * v[20];
	}
	if (i < (N-hh+1)) {
	    val += in[i+hh-1] * v[21];
	}
	if (i < (N-hh)) {
	    val += in[i+hh] * v[22];
	}
	if (i < (N-hh-1)) {
	    val += in[i+hh+1] * v[23];
	}
	if (i < (N-hh-2)) {
	    val += in[i+hh+2] * v[24];
	}

	out[i] = val;
	v += 25;
    }
}

int csrTo5x5(CSRmatrix *csr, matrix_5x5_t *m5x5)
{
    int i, j, offset;
    int h = m5x5->h;
    int hh = h + h;

    for (i = 0; i < csr->nrow; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    offset = csr->colIndex[j] - i;
	    if (offset == (-hh-2)) {
		m5x5->values[(i*25)] = csr->values[j];
	    }
	    else if (offset == (-hh-1)) {
		m5x5->values[(i*25)+1] = csr->values[j];
	    }
	    else if (offset == (-hh)) {
		m5x5->values[(i*25)+2] = csr->values[j];
	    }
	    else if (offset == (-hh+1)) {
		m5x5->values[(i*25)+3] = csr->values[j];
	    }
	    else if (offset == (-hh+2)) {
		m5x5->values[(i*25)+4] = csr->values[j];
	    }

	    else if (offset == (-h-2)) {
		m5x5->values[(i*25)+5] = csr->values[j];
	    }
	    else if (offset == (-h-1)) {
		m5x5->values[(i*25)+6] = csr->values[j];
	    }
	    else if (offset == (-h)) {
		m5x5->values[(i*25)+7] = csr->values[j];
	    }
	    else if (offset == (-h+1)) {
		m5x5->values[(i*25)+8] = csr->values[j];
	    }
	    else if (offset == (-h+2)) {
		m5x5->values[(i*25)+9] = csr->values[j];
	    }

	    else if (offset == -2) {
		m5x5->values[(i*25)+10] = csr->values[j];
	    }
	    else if (offset == -1) {
		m5x5->values[(i*25)+11] = csr->values[j];
	    }
	    else if (offset == 0) {
		m5x5->values[(i*25)+12] = csr->values[j];
	    }
	    else if (offset == 1) {
		m5x5->values[(i*25)+13] = csr->values[j];
	    }
	    else if (offset == 2) {
		m5x5->values[(i*25)+14] = csr->values[j];
	    }

	    else if (offset == (h-2)) {
		m5x5->values[(i*25)+15] = csr->values[j];
	    }
	    else if (offset == (h-1)) {
		m5x5->values[(i*25)+16] = csr->values[j];
	    }
	    else if (offset == (h)) {
		m5x5->values[(i*25)+17] = csr->values[j];
	    }
	    else if (offset == (h+1)) {
		m5x5->values[(i*25)+18] = csr->values[j];
	    }
	    else if (offset == (h+2)) {
		m5x5->values[(i*25)+19] = csr->values[j];
	    }

	    else if (offset == (hh-2)) {
		m5x5->values[(i*25)+20] = csr->values[j];
	    }
	    else if (offset == (hh-1)) {
		m5x5->values[(i*25)+21] = csr->values[j];
	    }
	    else if (offset == (hh)) {
		m5x5->values[(i*25)+22] = csr->values[j];
	    }
	    else if (offset == (hh+1)) {
		m5x5->values[(i*25)+23] = csr->values[j];
	    }
	    else if (offset == (hh+2)) {
		m5x5->values[(i*25)+24] = csr->values[j];
	    }
	    else {
		if (fabs(csr->values[j]) > 1e-12) {
		    /* not 5x5 banded */
		    return 0;
		}
	    }
	}
    }
    return 1;
}

void free5x5(matrix_5x5_t *mat)
{
    free(mat->values);
    free(mat);
}

#define INSERT_VALUE(x) \
    {\
    if (fabs(mat->values[(i*25)+x]) > 1e-15) {\
    result->values[idx] = mat->values[(i*25)+x];\
    result->colIndex[idx] = j;\
    idx++;\
    }\
    }\


CSRmatrix *m5x5ToCSR(matrix_5x5_t *mat)
{
    CSRmatrix *result;
    int i, j, idx;
    int h = mat->h;
    int hh = h + h;

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(result, mat->N, mat->N, mat->N*25);

    idx = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < mat->N; i++) {
	j = i - hh - 2;
	if (j >= 0) INSERT_VALUE(0);
	j = i - hh - 1;
	if (j >= 0) INSERT_VALUE(1);
	j = i - hh;
	if (j >= 0) INSERT_VALUE(2);
	j = i - hh + 1;
	if (j >= 0) INSERT_VALUE(3);
	j = i - hh + 2;
	if (j >= 0) INSERT_VALUE(4);

	j = i - h - 2;
	if (j >= 0) INSERT_VALUE(5);
	j = i - h - 1;
	if (j >= 0) INSERT_VALUE(6);
	j = i - h;
	if (j >= 0) INSERT_VALUE(7);
	j = i - h + 1;
	if (j >= 0) INSERT_VALUE(8);
	j = i - h + 2;
	if (j >= 0) INSERT_VALUE(9);

	j = i - 2;
	if (j >= 0) INSERT_VALUE(10);
	j = i - 1;
	if (j >= 0) INSERT_VALUE(11);
	j = i;
	INSERT_VALUE(12);
	j = i + 1;
	if (j < mat->N) INSERT_VALUE(13);
	j = i + 2;
	if (j < mat->N) INSERT_VALUE(14);

	j = i + h - 2;
	if (j < mat->N) INSERT_VALUE(15);
	j = i + h - 1;
	if (j < mat->N) INSERT_VALUE(16);
	j = i + h;
	if (j < mat->N) INSERT_VALUE(17);
	j = i + h + 1;
	if (j < mat->N) INSERT_VALUE(18);
	j = i + h + 2;
	if (j < mat->N) INSERT_VALUE(19);

	j = i + hh - 2;
	if (j < mat->N) INSERT_VALUE(20);
	j = i + hh - 1;
	if (j < mat->N) INSERT_VALUE(21);
	j = i + hh;
	if (j < mat->N) INSERT_VALUE(22);
	j = i + hh + 1;
	if (j < mat->N) INSERT_VALUE(23);
	j = i + hh + 2;
	if (j < mat->N) INSERT_VALUE(24);

	result->rowStart[i+1] = idx;
    }
    return result;
}

#undef INSERT_VALUE

void m5x5_matrix_add(matrix_5x5_t *in1, matrix_5x5_t *in2, matrix_5x5_t *out)
{
    int i;
    for (i = 0; i < (in1->N * 25); i++) {
	out->values[i] = in1->values[i] + in2->values[i];
    }
}

void m5x5_scalar_mult(matrix_5x5_t *mat, double scalar)
{
    int i;
    for (i = 0; i < (mat->N * 25); i++) {
	mat->values[i] *= scalar;
    }
}

matrix_3x3_t *allocate3x3(int N, int h)
{
    matrix_3x3_t *result;
    result = (matrix_3x3_t *)malloc(sizeof(matrix_3x3_t));
    if (!result) return NULL;

    result->N = N;
    result->h = h;

    result->values = (double *)malloc(N * 9 * sizeof(double));
    if (!result->values) {
	free(result);
	return NULL;
    }
    memset(result->values, 0, N * 9 * sizeof(double));

    return result;
}

matrix_3x3_t *m3x3_duplicate(matrix_3x3_t *mat)
{
    matrix_3x3_t *result = allocate3x3(mat->N, mat->h);
    memcpy(result->values, mat->values, mat->N * 9 * sizeof(double));
    return result;
}

void m3x3_vector_mult(matrix_3x3_t *m3x3, double *in, double *out)
{
    int i;
    double val;
    int h = m3x3->h;
    int N = m3x3->N;
    double *v;

    v = &m3x3->values[0];
    for (i = 0; i < (h+1); i++) {
	val = 0.0;
	if (i >= (h+1)) {
	    val += in[i-h-1] * v[0];
	}
	if (i >= (h)) {
	    val += in[i-h] * v[1];
	}
	if (i >= (h-1)) {
	    val += in[i-h+1] * v[2];
	}

	if (i >= 1) {
	    val += in[i-1] * v[3];
	}
	val += in[i] * v[4];
	if (i < (N-1)) {
	    val += in[i+1] * v[5];
	}
	
	if (i < (N-h+1)) {
	    val += in[i+h-1] * v[6];
	}
	if (i < (N-h)) {
	    val += in[i+h] * v[7];
	}
	if (i < (N-h-1)) {
	    val += in[i+h+1] * v[8];
	}

	out[i] = val;
	v += 9;
    }
    for (; i < (N-h-1); i++) {
	out[i] = in[i-h-1]*v[0] + in[i-h]*v[1] + in[i-h+1]*v[2]
	    + in[i-1]*v[3] + in[i]*v[4] + in[i+1]*v[5]
	    + in[i+h-1]*v[6] + in[i+h]*v[7] + in[i+h+1]*v[8];
	v += 9;
    }
    for (; i < N; i++) {
	val = 0.0;
	if (i >= (h+1)) {
	    val += in[i-h-1] * v[0];
	}
	if (i >= (h)) {
	    val += in[i-h] * v[1];
	}
	if (i >= (h-1)) {
	    val += in[i-h+1] * v[2];
	}

	if (i >= 1) {
	    val += in[i-1] * v[3];
	}
	val += in[i] * v[4];
	if (i < (N-1)) {
	    val += in[i+1] * v[5];
	}
	
	if (i < (N-h+1)) {
	    val += in[i+h-1] * v[6];
	}
	if (i < (N-h)) {
	    val += in[i+h] * v[7];
	}
	if (i < (N-h-1)) {
	    val += in[i+h+1] * v[8];
	}

	out[i] = val;
	v += 9;
    }
}

int csrTo3x3(CSRmatrix *csr, matrix_3x3_t *m3x3)
{
    int i, j, offset;
    int h = m3x3->h;

    for (i = 0; i < csr->nrow; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    offset = csr->colIndex[j] - i;

	    if (offset == (-h-1)) {
		m3x3->values[(i*9)+0] = csr->values[j];
	    }
	    else if (offset == (-h)) {
		m3x3->values[(i*9)+1] = csr->values[j];
	    }
	    else if (offset == (-h+1)) {
		m3x3->values[(i*9)+2] = csr->values[j];
	    }

	    else if (offset == -1) {
		m3x3->values[(i*9)+3] = csr->values[j];
	    }
	    else if (offset == 0) {
		m3x3->values[(i*9)+4] = csr->values[j];
	    }
	    else if (offset == 1) {
		m3x3->values[(i*9)+5] = csr->values[j];
	    }

	    else if (offset == (h-1)) {
		m3x3->values[(i*9)+6] = csr->values[j];
	    }
	    else if (offset == (h)) {
		m3x3->values[(i*9)+7] = csr->values[j];
	    }
	    else if (offset == (h+1)) {
		m3x3->values[(i*9)+8] = csr->values[j];
	    }
	    else {
		if (fabs(csr->values[j]) > 1e-12) {
		    /* not 3x3 banded */
		    return 0;
		}
	    }
	}
    }
    return 1;
}

void free3x3(matrix_3x3_t *mat)
{
    free(mat->values);
    free(mat);
}

#define INSERT_VALUE(x) \
    {\
    if (fabs(mat->values[(i*9)+x]) > 0) {\
    result->values[idx] = mat->values[(i*9)+x];\
    result->colIndex[idx] = j;\
    idx++;\
    }\
    }\

CSRmatrix *m3x3ToCSR(matrix_3x3_t *mat)
{
    CSRmatrix *result;
    int i, j, idx;
    int h = mat->h;

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    CSR_setup(result, mat->N, mat->N, mat->N*9);

    idx = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < mat->N; i++) {
	j = i - h - 1;
	if (j >= 0) INSERT_VALUE(0);
	j = i - h;
	if (j >= 0) INSERT_VALUE(1);
	j = i - h + 1;
	if (j >= 0) INSERT_VALUE(2);
	j = i - 1;
	if (j >= 0) INSERT_VALUE(3);
	j = i;
	INSERT_VALUE(4);
	j = i + 1;
	if (j < mat->N) INSERT_VALUE(5);
	j = i + h - 1;
	if (j < mat->N) INSERT_VALUE(6);
	j = i + h;
	if (j < mat->N) INSERT_VALUE(7);
	j = i + h + 1;
	if (j < mat->N) INSERT_VALUE(8);

	result->rowStart[i+1] = idx;
    }
    return result;
}

#define OUTVAL(x) out->values[(i*25) + x]

#define IN1_0 in1->values[(i*9)+0]
#define IN1_1 in1->values[(i*9)+1]
#define IN1_2 in1->values[(i*9)+2]
#define IN1_3 in1->values[(i*9)+3]
#define IN1_4 in1->values[(i*9)+4]
#define IN1_5 in1->values[(i*9)+5]
#define IN1_6 in1->values[(i*9)+6]
#define IN1_7 in1->values[(i*9)+7]
#define IN1_8 in1->values[(i*9)+8]

#define IN2_0 ((j+h+1) >= N || (j+h+1) < 0 ? 0.0 : in2->values[((j+h+1)*9)])
#define IN2_1 ((j+h) >= N || (j+h) < 0 ? 0.0 : in2->values[((j+h)*9)+1])
#define IN2_2 ((j+h-1) >= N || (j+h-1) < 0 ? 0.0 : in2->values[((j+h-1)*9)+2])
#define IN2_3 ((j+1) >= N || (j+1) < 0 ? 0.0 : in2->values[((j+1)*9)+3])
#define IN2_4 (j >= N || j < 0 ? 0.0 : in2->values[(j*9)+4])
#define IN2_5 ((j-1) >= N || (j-1) < 0 ? 0.0 : in2->values[((j-1)*9)+5])
#define IN2_6 ((j-h+1) >= N || (j-h+1) < 0 ? 0.0 : in2->values[((j-h+1)*9)+6])
#define IN2_7 ((j-h) >= N || (j-h) < 0 ? 0.0 : in2->values[((j-h)*9)+7])
#define IN2_8 ((j-h-1) >= N || (j-h-1) < 0 ? 0.0 : in2->values[((j-h-1)*9)+8])

void m3x3_matrix_multiply(matrix_3x3_t *in1, matrix_3x3_t *in2, matrix_5x5_t *out)
{
    int i, j, k;
    int N = out->N;
    int h = out->h;
    int hh = h + h;

    for (i = 0; i < N; i++) {
	j = i - hh - 2;
	if (j >= 0) OUTVAL(0) = IN1_0*IN2_0;
	j = i - hh - 1;
	if (j >= 0) OUTVAL(1) = IN1_0*IN2_1 + IN1_1*IN2_0;
	j = i - hh;
	if (j >= 0) OUTVAL(2) = IN1_0*IN2_2 + IN1_1*IN2_1 + IN1_2*IN2_0;
	j = i - hh + 1;
	if (j >= 0) OUTVAL(3) = IN1_1*IN2_2 + IN1_2*IN2_1;
	j = i - hh + 2;
	if (j >= 0) OUTVAL(4) = IN1_2*IN2_2;

	j = i - h - 2;
	if (j >= 0) OUTVAL(5) = IN1_0*IN2_3 + IN1_3*IN2_0;
	j = i - h - 1;
	if (j >= 0) OUTVAL(6) = IN1_0*IN2_4 + IN1_1*IN2_3 + IN1_3*IN2_1 + IN1_4*IN2_0;
	j = i - h;
	if (j >= 0) OUTVAL(7) = IN1_0*IN2_5 + IN1_1*IN2_4 + IN1_2*IN2_3 +
			IN1_3*IN2_2 + IN1_4*IN2_1 + IN1_5*IN2_0;
	j = i - h + 1;
	if (j >= 0) OUTVAL(8) = IN1_1*IN2_5 + IN1_2*IN2_4 + IN1_4*IN2_2 + IN1_5*IN2_1;
	j = i - h + 2;
	if (j >= 0) OUTVAL(9) = IN1_2*IN2_5 + IN1_5*IN2_2;

	j = i - 2;
	if (j >= 0) OUTVAL(10) = IN1_0*IN2_6 + IN1_3*IN2_3 + IN1_6*IN2_0;
	j = i - 1;
	if (j >= 0) OUTVAL(11) = IN1_0*IN2_7 + IN1_1*IN2_6 + IN1_3*IN2_4 +
			IN1_4*IN2_3 + IN1_6*IN2_1 + IN1_7*IN2_0;
	j = i;
	OUTVAL(12) = IN1_0*IN2_8 + IN1_1*IN2_7 + IN1_2*IN2_6 +
	    IN1_3*IN2_5 + IN1_4*IN2_4 + IN1_5*IN2_3 +
	    IN1_6*IN2_2 + IN1_7*IN2_1 + IN1_8*IN2_0;
	j = i + 1;
	if (j < N) OUTVAL(13) = IN1_1*IN2_8 + IN1_2*IN2_7 + IN1_4*IN2_5 +
		       IN1_5*IN2_4 + IN1_7*IN2_2 + IN1_8*IN2_1;
	j = i + 2;
	if (j < N) OUTVAL(14) = IN1_2*IN2_8 + IN1_5*IN2_5 + IN1_8*IN2_2;

	j = i + h - 2;
	if (j < N) OUTVAL(15) = IN1_3*IN2_6 + IN1_6*IN2_3;
	j = i + h - 1;
	if (j < N) OUTVAL(16) = IN1_3*IN2_7 + IN1_4*IN2_6 + IN1_6*IN2_4 + IN1_7*IN2_3;
	j = i + h;
	if (j < N) OUTVAL(17) = IN1_3*IN2_8 + IN1_4*IN2_7 + IN1_5*IN2_6 +
		       IN1_6*IN2_5 + IN1_7*IN2_4 + IN1_8*IN2_3;
	j = i + h + 1;
	if (j < N) OUTVAL(18) = IN1_4*IN2_8 + IN1_5*IN2_7 + IN1_7*IN2_5 + IN1_8*IN2_4;
	j = i + h + 2;
	if (j < N) OUTVAL(19) = IN1_5*IN2_8 + IN1_8*IN2_5;

	j = i + hh - 2;
	if (j < N) OUTVAL(20) = IN1_6*IN2_6;
	j = i + hh - 1;
	if (j < N) OUTVAL(21) = IN1_6*IN2_7 + IN1_7*IN2_6;
	j = i + hh;
	if (j < N) OUTVAL(22) = IN1_6*IN2_8 + IN1_7*IN2_7 + IN1_8*IN2_6;
	j = i + hh + 1;
	if (j < N) OUTVAL(23) = IN1_7*IN2_8 + IN1_8*IN2_7;
	j = i + hh + 2;
	if (j < N) OUTVAL(24) = IN1_8*IN2_8;
    }
}

void m3x3_matrix_add(matrix_3x3_t *in1, matrix_3x3_t *in2, matrix_3x3_t *out)
{
    int i;
    for (i = 0; i < (in1->N * 9); i++) {
	out->values[i] = in1->values[i] + in2->values[i];
    }
}

void m3x3_transpose(matrix_3x3_t *in, matrix_3x3_t *out)
{
    int i;
    int idx, idxT;
    int h = in->h;
    int N = in->N;

    idx = 0;
    for (i = 0; i < N; i++) {
	/* -h-1 */
	if (i >= (h+1)) {
	    idxT = ((i-h-1)*9)+8;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* -h */
	if (i >= h) {
	    idxT = ((i-h)*9)+7;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* -h+1 */
	if (i >= (h-1)) {
	    idxT = ((i-h+1)*9)+6;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* -1 */
	if (i >= 1) {
	    idxT = ((i-1)*9)+5;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* diagonal is same */
	out->values[idx] = in->values[idx];
	idx++;

	/* +1 */
	if (i < (N-1)) {
	    idxT = ((i+1)*9)+3;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* +h-1 */
	if (i < (N-h+1)) {
	    idxT = ((i+h-1)*9)+2;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* +h */
	if (i < (N-h)) {
	    idxT = ((i+h)*9)+1;
	    out->values[idxT] = in->values[idx];
	}
	idx++;

	/* +h+1 */
	if (i < (N-h-1)) {
	    idxT = ((i+h+1)*9)+0;
	    out->values[idxT] = in->values[idx];
	}
	idx++;
    }
}

void m3x3_scalar_mult(matrix_3x3_t *mat, double scalar)
{
    int i;
    for (i = 0; i < (mat->N * 9); i++) {
	mat->values[i] *= scalar;
    }
}
