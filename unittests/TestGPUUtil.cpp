/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestGPUUtil.h"

#include "GPUUtil.h"

extern "C" {
#include "csrmatrix.h"
};

void TestGPUUtil::setUp()
{
}

void TestGPUUtil::tearDown()
{
}

void TestGPUUtil::testGPUUtil()
{
    int i;
    // create a matrix
    int npl1 = 100;
    CSRmatrix *cpumat;

    cpumat = CSR_create_eye(npl1);
    CSR_scalar_mult(cpumat, 2.5);
    
    double *x, *b, *d_x, *d_b;

    // create vectors
    x = new double[npl1];
    b = new double[npl1];
    for (i = 0; i < npl1; i++) {
	x[i] = (double)i;
	b[i] = 0.0;
    }

    // create vectors on GPU
    d_x = (double *)cudaMallocNess(npl1 * sizeof(double));
    d_b = (double *)cudaMallocNess(npl1 * sizeof(double));
    cudaMemcpyH2D(d_x, x, npl1 * sizeof(double));

    // perform matrix multiply on CPU
    CSR_matrix_vector_mult(cpumat, x, b);

    // copy matrix to GPU
    GpuMatrix_t *gpumat = csrToGpuMatrix(cpumat);
    
    // perform matrix multiply on GPU
    gpuMatrixMultiply(gpumat, d_x, d_b);

    // copy result back to x array and check it
    cudaMemcpyD2H(x, d_b, npl1 * sizeof(double));

    for (i = 0; i < npl1; i++) {
	CPPUNIT_ASSERT_DOUBLES_EQUAL(b[i], x[i], 1e-10);
    }
    
    // free everything
    freeGpuMatrix(gpumat);
    cudaFreeNess(d_x);
    cudaFreeNess(d_b);
    CSR_free(cpumat);
}

