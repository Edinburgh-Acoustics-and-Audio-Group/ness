#ifndef _BANDED_H_
#define _BANDED_H_

#include "csrmatrix.h"

/*
 * A matrix that implements a 5x5 stencil across a plate
 */
typedef struct matrix_5x5_s
{
    int N; /* size in points */
    int h; /* height in points of the plate */
    double *values;
} matrix_5x5_t;

matrix_5x5_t *allocate5x5(int N, int h);
void m5x5_vector_mult(matrix_5x5_t *m5x5, double *in, double *out);
int csrTo5x5(CSRmatrix *csr, matrix_5x5_t *m5x5);
void free5x5(matrix_5x5_t *mat);

CSRmatrix *m5x5ToCSR(matrix_5x5_t *mat);
void m5x5_matrix_add(matrix_5x5_t *in1, matrix_5x5_t *in2, matrix_5x5_t *out);
void m5x5_scalar_mult(matrix_5x5_t *mat, double scalar);
matrix_5x5_t *m5x5_duplicate(matrix_5x5_t *mat);

typedef struct matrix_3x3_s
{
    int N; /* size in points */
    int h; /* height in points of the plate */
    double *values;
} matrix_3x3_t;

matrix_3x3_t *allocate3x3(int N, int h);
void m3x3_vector_mult(matrix_3x3_t *m3x3, double *in, double *out);
int csrTo3x3(CSRmatrix *csr, matrix_3x3_t *m3x3);
void free3x3(matrix_3x3_t *mat);

CSRmatrix *m3x3ToCSR(matrix_3x3_t *mat);
void m3x3_matrix_multiply(matrix_3x3_t *in1, matrix_3x3_t *in2, matrix_5x5_t *out);
void m3x3_matrix_add(matrix_3x3_t *in1, matrix_3x3_t *in2, matrix_3x3_t *out);
void m3x3_transpose(matrix_3x3_t *in, matrix_3x3_t *out);
void m3x3_scalar_mult(matrix_3x3_t *mat, double scalar);
matrix_3x3_t *m3x3_duplicate(matrix_3x3_t *mat);

#ifdef __cplusplus
extern "C" {
#endif
void m3x3_vector_SSE(matrix_3x3_t *m3x3, double *in, double *out);
void m5x5_vector_SSE(matrix_5x5_t *m5x5, double *in, double *out);
#ifdef __cplusplus
};
#endif

#endif
