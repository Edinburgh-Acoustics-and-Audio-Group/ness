/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 *
 * AVX kernels for modal plate.
 */

#include "ModalPlate.h"

#ifdef USE_AVX

#include <immintrin.h>

// H1 must point to the H1 vals required for these 4 rows, not to the very
// beginning of H1vals
void ModalPlate::avxUpdate4Rows(double *q1, double *H1, double *result,
				int *q1index, int nnz)
{
    int i;

    // clear running total
    __m256d sum = _mm256_setzero_pd();

    // loop over all non-zeroes
    for (i = 0; i < nnz; i++) {
	// broadcast from q1
	__m256d q1vals = _mm256_broadcast_sd(&q1[q1index[i]]);

	// load 4 values from H1
	__m256d H1vals = _mm256_loadu_pd(&H1[i*4]);

	// multiply and add
	__m256d prod = _mm256_mul_pd(q1vals, H1vals);
	sum = _mm256_add_pd(sum, prod);
    }

    // store result
    _mm256_storeu_pd(result, sum);
}

void ModalPlate::avxUpdateSingleRow(double *q1, double *H1, double *result,
				    int *q1index, int nnz)
{
    int i;

    // clear running total
    double sum = 0.0;

    // loop over all non-zeroes
    for (i = 0; i < nnz; i++) {
	sum += (q1[q1index[i]] * H1[i]);
    }
    *result = sum;
}

#endif
