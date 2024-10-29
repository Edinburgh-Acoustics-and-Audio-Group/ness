/**
 * Originally written by Savvas Petrou.
 * Extended by James Perry and Adrian Mouat.
 * Copyright (c) 2010, 2012 The University of Edinburgh
 */

#ifndef CSRMATRIX_H
#define CSRMATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include "arralloc.h"

#ifdef SINGLE
    #define real float
#else
    #define real double
#endif

#define TRUE  1
#define FALSE 0

/**
  * Had to start ops at 5 and duplicate code to avoid issue where Savvas used
  * different numbers in different functions.
  */
enum op_enum {OP_ADD = 5, OP_SUB, OP_MULT, OP_DIV};

/* struct for CSR matrix type */
typedef struct
{
    int    nrow;
    int    ncol;
    int    nzmax;
    int   *colIndex;
    int   *rowStart;
    real  *values;
} CSRmatrix;

/* struct for CSC matrix type */
typedef struct
{
    int    nrow;
    int    ncol;
    int    nzmax;
    int   *rowIndex;
    int   *colStart;
    real  *values;
} CSCmatrix;


/* =================== * 
 *    CSR functions    * 
 * =================== */

void CSRPrint(CSRmatrix* csrmatrix, char* name);

// Allocate memory needed by the sparse matrix
int CSR_setup(CSRmatrix *csr, int r, int c, int nzmax);

// Sets a single value in the matrix
void CSRSetValue(CSRmatrix *csr, int row, int col, real val);

// Sets a whole row of values to zero
void CSR_zero_row(CSRmatrix *csr, int row);

// Sets a whole column of values to zero
void CSR_zero_column(CSRmatrix *csr, int col);

// Gets a single value from the matrix
real CSRGetValue(CSRmatrix *csr, int row, int col);

// Print out the full sparse matrix (including zeros)
void CSRPrint(CSRmatrix* csrmatrix, char* name); 

// Print a CSR matrix as a full matrix
void CSRFullPrint(CSRmatrix *csr, char *filename);

// Print a CSR matrix to match Matlab output
void CSRMatlabPrint(CSRmatrix *csr, char *filename);

// ----------------------------------------------------------------------------------------
// -  Perform a matrix vector operation                                                   -
// -                                                                                      -
// -  If you set "init_vec" the elements of the result vector are initialized to zero     -
// -  When (op == 1) and (init_vec == 1) then its normal matrix-vector multiplication     -
// -  and the result is stored in b.                                                      -
// -                                                                                      -
// -  But you can do more with this function. When (init_vec==0) then the b vector        -
// -  can be used as the result of a previous computation.                                -
// -  For example:                                                                        -
// -    CSR_matrix_vector(A, v, b, 1, 1); --> This will compute A*v and put it in b.      -
// -    CSR_matrix_vector(B, h, b, 0, 2); --> This will actually result to (A*v) - (B*h)  -
// -    ....                                                                              -
// -                                                                                      -
// -  Based on the same logic the other operations can be used (multiply and divide)      -
// ----------------------------------------------------------------------------------------
/*
 * WARNING: the multiplication and division options here don't give the result
 * you might expect (i.e. not the same as doing the matrix multiply into a
 * separate buffer and then explicitly multiplying/dividing) so should probably
 * be avoided.
 */
void CSR_matrix_vector(CSRmatrix *in_CSR, real *v, real *b, int init_vec, int op);

/**
 * Convienience function that calls matrix vector for normal case and returns result vector.
 */
real *CSR_matrix_vector_mult(CSRmatrix* csr, real* v, real* r); 

// Perform a scalar computation on all elements of the sparse matrix
void CSR_scalar_computation(CSRmatrix *in_CSR, real val, int comp);

//Note following are in-place although also return input matrix to support
//chaining
CSRmatrix* CSR_scalar_mult(CSRmatrix *in_CSR, real val);

CSRmatrix* CSR_scalar_div(CSRmatrix *in_CSR, real val);

// Contruct a Symmetric Toeplitz sparse matrix
CSRmatrix *CSR_sym_toeplitz(real *in_row, int row_len);

//Construct a non-symmetric toeplitz matrix
//Column wins disagreement
//Order agrees with matlab
CSRmatrix *CSR_toeplitz(real *in_col, int col_len, real *in_row, int row_len);

// Duplicate the sparse matrix
CSRmatrix *CSR_duplicate(CSRmatrix *in_CSR);

// Free all allocated memory
int CSR_free(CSRmatrix *in_CSR);
int CSC_free(CSCmatrix *in_CSC);

// Copies a matrix, re-using the original structure of the output matrixx
void CSR_copy(CSRmatrix *in_CSR, CSRmatrix *out_CSR);

void CSR_shallow_copy(CSRmatrix* copy, CSRmatrix* orig);

// Kronecker tensor product of a Symmetric Toeplitz Sparse matrix and an identity matrix
CSRmatrix *CSR_kron_mat_eye(CSRmatrix *in_CSR, int eye_rank);

CSRmatrix *CSR_kron_toeplitz_eye(real *in_vec, int vec_len, int eye_rank);

// Kronecker tensor product of an identity matrix and a Symmetric Toeplitz Sparse matrix
CSRmatrix *CSR_kron_eye_mat(CSRmatrix *in_CSR, int eye_rank);

// Kronecker product of any two sparse matrices
CSRmatrix *CSR_kronecker_product(CSRmatrix *a, CSRmatrix *b);

// Populate the full matrix from a sparse one
real **CSR_get_full(CSRmatrix *in_CSR);

// Create a CSR matrix from a full matrix
CSRmatrix *full_to_CSR(real *in_mat, int rows, int cols);

// Output the sparse matrix as an PGM image
void CSR_PGM_output(CSRmatrix *in_CSR, char *filename);

// Raise a sparce CRS matrix in square
CSRmatrix *CSR_matrix_square(CSRmatrix *in_CSR);

// Multiplies two CSR matrices
CSRmatrix *CSR_matrix_multiply(CSRmatrix *in_CSR_1, CSRmatrix *in_CSR_2);

// Multiplies two CSR matrices, reusing non-zero structure
void CSR_matrix_multiply_reuse(CSRmatrix *in_CSR1, CSRmatrix *in_CSR2, CSRmatrix *out_CSR);

// Add/subtract two CSR matrices
CSRmatrix *CSR_matrix_add_sub(CSRmatrix *in_CSR_1, CSRmatrix *in_CSR_2, int add_sub);

//Avoid need to pass const for adding
CSRmatrix *CSR_matrix_add(CSRmatrix *in_CSR1, CSRmatrix *in_CSR2);

//And subtracting
CSRmatrix *CSR_matrix_sub(CSRmatrix* in_CSR1, CSRmatrix* in_CSR2);

// Add two CSR matrices but reuse a previous result matrix structure
void CSR_matrix_add_reuse(CSRmatrix* in_CSR1, CSRmatrix *in_CSR2, CSRmatrix *out_CSR);

// Function that creates square identity matrices
CSRmatrix *CSR_create_eye(int eye_rank);

CSRmatrix *CSR_transpose(CSRmatrix *in_CSR);

// Remove selected rows from a matrix
int CSR_cut_rows(CSRmatrix *csr, int *rowsToKeep, int len);

// Remove selected columns from a matrix
int CSR_cut_cols(CSRmatrix *csr, int *colsToKeep, int len);

// Zero selected rows in a matrix
void CSR_zero_rows(CSRmatrix *csr, int *rowsToKeep, int len);

// Zero selected columns in a matrix
void CSR_zero_cols(CSRmatrix *csr, int *colsToKeep, int len);

// Scale the rows of the matrix by values from the vector
void CSR_diagonal_scale(CSRmatrix *csr, double *vec);

// Scale the columns of the matrix by values from the vector
void CSR_column_scale(CSRmatrix *csr, double *vec);

// Extract a sub-matrix from an existing matrix
CSRmatrix *CSR_get_sub_matrix(CSRmatrix *csr, int row, int col, int height, int width);

// Saves a sparse matrix to PETSc binary format
int CSR_save_petsc(char *filename, CSRmatrix *csr);

// Load a PETSc format binary matrix file
CSRmatrix *CSR_load_petsc(char *filename);

// Compares two matrices for equality within a specified tolerance
int CSR_compare(CSRmatrix *csr1, CSRmatrix *csr2, double tolerance);

// Faster (but less flexible) matrix-by-vector multiply
void CSR_fast_mat_vec_mult(CSRmatrix* A, double* x, double* b);

// Creates a sparse block diagonal matrix consisting of A and B combine diagonally
CSRmatrix* CSR_blk_diag(CSRmatrix *A, CSRmatrix* B);

// Kronecker product of A (x) B where A is actually a diagonal matrix
CSRmatrix *CSR_kron_diag_mat(CSRmatrix *in_CSR, double* diag, int diag_size);

// Multiplies two sparse matrices which are square and only populated on the diagonal
CSRmatrix *CSR_matrix_mult_sqr_diag(CSRmatrix* in_CSR_1, CSRmatrix* in_CSR_2);

// Selectively zero rows in a sparse matrix
CSRmatrix* CSR_zero_rows_new(CSRmatrix *csr, int *rowsToKeep);


/* =================== * 
 *    CSC functions    * 
 * =================== */

// Allocate memory needed by the sparse matrix
int CSC_setup(CSCmatrix *csc, int r, int c, int nzmax); 

// Print out the full sparse matrix (including zeros)
void CSC_print(CSCmatrix *csc, char *filename);

//Convert a sparse matrix in CSR form to CSC form
CSCmatrix *convert_CSR_to_CSC(CSRmatrix *in_CSR);

#endif
