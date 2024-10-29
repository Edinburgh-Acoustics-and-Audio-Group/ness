/**
 * Originally written by Savvas Petrou.
 * Extended by James Perry and Adrian Mouat.
 * Copyright (c) 2010, 2012 The University of Edinburgh
 */
#include "csrmatrix.h"
#include <assert.h>
 
#ifndef DEBUG
#define DEBUG 1
#endif

// Internal function to check if sparse matrix is valid
int CSR_valid_check(CSRmatrix *in_CSR);

// Internal function to extend the memory of a CSR matrix
int CSR_extend(CSRmatrix *in_CSR);
/* =================== * 
 *    CSR functions    * 
 * =================== */

/* ***************************************** *
 *  Function to allocate memory for the CSR  *
 *  matrix.                                  *
 *                                           *
 *  csr   : the matrix to be setup            *
 *  r     : number of rows                    *
 *  c     : number of columns                 *
 *  nzmax : max number of non-zero entries    *
 * ***************************************** */
int CSR_setup(CSRmatrix *csr, int r, int c, int nzmax) 
{
    long long max_nzmax;

    csr->nrow=r;
    csr->ncol=c;
    
    // Check if requested non_zero elements are more than the maximum (full matrix)
    // This can overflow a 32-bit int in some cases
    max_nzmax = (long long)r * (long long)c;
    if ( max_nzmax < nzmax ) {
        csr->nzmax = max_nzmax;
    } else {
        csr->nzmax = nzmax;
    }

    //AM: Removed commented out calls to malloc (presumably changed to calloc
    //to avoid needing to zero array (not convinced this is portable though!)
    csr->colIndex = (int *)calloc(nzmax, sizeof(int));
    csr->rowStart = (int *)calloc(r+1, sizeof(int));
    csr->values   = (real *)calloc(nzmax, sizeof(real));

    // Check if all memory was allocated successfully
    if ( csr->values == NULL 
	 || csr->colIndex == NULL 
	 || csr->rowStart == NULL ) {
        printf("\nCSR matrix memory allocation failed.\n");

        // Free any memory allocated
        if (csr->values != NULL) free(csr->values);
        if (csr->colIndex != NULL) free(csr->colIndex);
        if (csr->rowStart != NULL) free(csr->rowStart);

        return 0;
    }

    // All OK
    return 1;
}

/**
 * CSRGetValue - Return the value at a given row and column in a sparse
 * matrix.
 * @csr The sparse matrix
 * @row Row index to get
 * @col Col index to get
 */
real CSRGetValue(CSRmatrix *csr, int row, int col)
{
    int i;

    for (i = csr->rowStart[row]; i < csr->rowStart[row+1]; i++) {
	if (csr->colIndex[i] == col) {
	    return csr->values[i];
	} else if (csr->colIndex[i] > col) {
	    break;
	}
    }

    return 0.0;
}


/* ****************************************** *
 *  Print out a CSR matrix as a full matrix.  *
 * ****************************************** */
void CSRFullPrint(CSRmatrix *csr, char *filename)
{
    int count = 0;
    int row, col;
    FILE *stream;

    // Check if filename passed as argument.
    // If yes then open the stream and write everything there.
    // If no then output to standard output.
    if ( filename == NULL ) {
        stream = stdout;
    } else {
        stream = fopen(filename, "w");
        if ( stream == NULL ) {
            printf("\nCSRFullPrint: Unable to open file. Function will print in stdout.\n\n");
            stream = stdout;
        }
    }

    fprintf(stream, "\n\nCSR Library: Printing sparse matrix");
    fprintf(stream, "\n-----------------------------------\n\n");

    for(row=0; row < csr->nrow; row++) {
        for(col=0; col < csr->ncol; col++) {
            if( (csr->colIndex[count] == col) && (count < csr->rowStart[row+1]) ) {
                fprintf(stream, "%9.4lf  ", csr->values[count]);
                count++;
            }
            else {
                //printf("%9.4lf  ", 0.0);
                fprintf(stream, "%9d  ", 0);
            }
        }
        fprintf(stream, "\n");
    }

    // If file stream opened then close it
    if (stream != stdout) {
        fclose(stream);
    }
}

  /*
   * CSRMatlabPrint - print out a CSR matrix in similar form to Matlab.
   *
   * Prints in column order, numbering elements from 1
   */
void CSRMatlabPrint(CSRmatrix *csr, char *filename)
{
    int count = 0;
    int row, col;
    FILE *stream;

    // Check if filename passed as argument.
    // If yes then open the stream and write everything there.
    // If no then output to standard output.
    if ( filename == NULL ) {
        stream = stdout;
    } else {
        stream = fopen(filename, "w");
        if ( stream == NULL ) {
            printf("\nCSRMatlablPrint: Unable to open file. \
                    Function will print to stdout.\n\n");
            stream = stdout;
        }
    }

    fprintf(stream, "\n\nCSR Library: Printing sparse matrix");
    fprintf(stream, "\n-----------------------------------\n\n");

    //Now do something weird. Matlab orders output by column, to do the same
    //we need to transpose!
    CSRmatrix* temp = CSR_transpose(csr);

    for(row=0; row < temp->nrow; row++) {
        for(col=0; col < temp->ncol; col++) {
            if( (temp->colIndex[count] == col) && (count < temp->rowStart[row+1]) ) {

		if ((temp->values[count] > 1e-20) || (temp->values[count] < -1e-20)) {
		    fprintf(stream, "(%d,%d)  %.20lf\n", 
			    col+1, row+1, temp->values[count]);
		}
		count++;
            }
        }
    }

    CSR_free(temp);
    // If file stream opened then close it
    if (stream != stdout) {
        fclose(stream);
    }
}

void CSRPrint(CSRmatrix* csrmatrix, char* name) {
    int i;
    printf("NDB CSRPrint(%s) nrow=%d ncol=%d nzmaz=%d ",name,
	   csrmatrix->nrow,
	   csrmatrix->ncol,
	   csrmatrix->nzmax);
    printf("colIndex=[");
    for (i = 0; i < csrmatrix->nzmax; i++) {
	printf("%d ", csrmatrix->colIndex[i]);
    }
    printf("]");
    printf(" rowStart=[");
    for (i = 0; i <= csrmatrix->nrow; i++) {
	printf("%d ", csrmatrix->rowStart[i]);
    }
    printf("]");
    printf("  values=[");
    for (i = 0; i < csrmatrix->nzmax; i++) {
	printf("%.16f ", csrmatrix->values[i]);
    }
    printf("]\n");
}


/* *********************************** *
 *  Populate the complete full matrix  *
 * *********************************** */
real **CSR_get_full(CSRmatrix *in_CSR)
{
    real **result;
    int row, col, count=0;

    // Allocate space for the full matrix
    result = (real **)arralloc(sizeof(real), 2, in_CSR->nrow, in_CSR->ncol);

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_get_full: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Fill in the full matrix
    for(row=0; row < in_CSR->nrow; row++) {
        for(col=0; col < in_CSR->ncol; col++) {
            if( (in_CSR->colIndex[count] == col) && (count < in_CSR->rowStart[row+1]) ) {
                result[row][col] = in_CSR->values[count];
                count++;
            }
            else {
                result[row][col] = 0.0;
            }
        }
    }

    return(result);
}


/* **************************************** *
 *  Create a CSR matrix from a full matrix  *
 * **************************************** */
CSRmatrix *full_to_CSR(real *in_mat, int rows, int cols)
{
    CSRmatrix *result;
    int i, j, nzmax, count = 0;

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nfull_to_CSR: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Set nzmax to 20% of the rows of the matrix
    nzmax = (int)ceil(rows*0.2) * cols;

    if( !CSR_setup(result, rows, cols, nzmax) ) {
        printf("\nfull_to_CSR: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    for(i=0; i < rows; i++) {

        // Check if there is enough memory to store a full row.
        // If not reallocate.
        if ( (result->nzmax - result->rowStart[i]) < cols )
            if( !CSR_extend(result) ) {
                printf("\nfull_to_CSR: Not enough memory for sparse matrix.\n");
                return(NULL);
            }

        // Add all non-zero elements for all columns in the current row
        for(j=0; j < cols; j++) {
            if( in_mat[i*cols+j] != 0.0 ) {
                result->values[count] = in_mat[i*cols+j];
                result->colIndex[count++] = j;
                result->rowStart[i+1]++;
            }
        }
        result->rowStart[i+1] += result->rowStart[i];
    }

    return(result);
}


/* ************************** *
 *  Prints out a full matrix  *
 * ************************** */
void print_full(real *in_mat, int nrow, int ncol, char *filename)
{
    int count = 0;
    int i, j;
    FILE *stream;

    // Check if filename passed as argument.
    // If yes then open the stream and write everything there.
    // If no then output to standard output.
    if ( filename == NULL )
        stream = stdout;
    else {
        stream = fopen(filename, "w");
        if ( stream == NULL ) {
            printf("\nprint_full: Unable to open file. Function will print in stdout.\n\n");
            stream = stdout;
        }
    }

    fprintf(stream, "\n\nCSR Library: Printing full matrix");
    fprintf(stream, "\n---------------------------------\n\n");

    for(i=0; i < nrow; i++) {
        for(j=0; j < ncol; j++) {
            if( in_mat[count] != 0.0 ) {
                fprintf(stream, "%9.4lf  ", in_mat[count]);
            } else {
                fprintf(stream, "%9d  ", 0);
            }
            count++;
        }
        fprintf(stream, "\n");
    }

    // If file stream opened then close it
    if ( stream != stdout )
        fclose(stream);
}

/**
 * CSR_zero_row - clear a whole row of the matrix to zero
 */
void CSR_zero_row(CSRmatrix *csr, int row)
{
    int i;
    for (i = csr->rowStart[row]; i < csr->rowStart[row+1]; i++) {
	csr->values[i] = 0.0;
    }
}

void CSR_zero_column(CSRmatrix *csr, int col)
{
    int i;
    for (i = 0; i < csr->rowStart[csr->nrow]; i++) {
	if (csr->colIndex[i] == col) {
	    csr->values[i] = 0.0;
	}
    }
}

/**
 * CSRSetValue - Set an element in a sparse matrix
 * @csr - The sparse matrix to update
 * @row - The row the element is on
 * @col - The col the element is on
 * @val - The value for the element
 */
void CSRSetValue(CSRmatrix *csr, int row, int col, real val)
{
    assert(col < csr->ncol);
    assert(row < csr->nrow);

    int nrow = csr->nrow;
    int rs, re, newCol, tmpCol, i = 0;
    int done = FALSE;
    real tmpVal, newVal = 0.0;

    rs = csr->rowStart[row];
    re = csr->rowStart[row+1];
    i = rs;

    if (re > rs) {
        //Row exists
        for (i = rs; i < re; i++) {
            if (csr->colIndex[i] == col) {
                //Luck!
                //Value exists, so just update
                csr->values[i] = val;
                done = TRUE;
                break;
            } else if (csr->colIndex[i] > col) {
                break;
            }
        }
    }
    if (!done) {

        if ((csr->nzmax - csr->rowStart[csr->nrow]) < csr->nrow) {
            if( !CSR_extend(csr) ) {
                printf("\nCSR_set_element: Not enough memory\n");
                return;
            }
        }

        newCol = col;
        newVal = val;
        while (i < csr->rowStart[csr->nrow]) {
            //store current values
            tmpVal = csr->values[i];
            tmpCol = csr->colIndex[i];

            csr->values[i] = newVal;
            csr->colIndex[i] = newCol;

            newCol = tmpCol;
            newVal = tmpVal;
            i++;
        }
        csr->values[i] = newVal;
        csr->colIndex[i] = newCol;

        //Now sort out row
        for (i = (row + 1); i <= nrow; i++) {
	    csr->rowStart[i]++; 
        }
    }
}

/**
  * CSR_matrix_vector_mult - Helper function does matrix vector multiplication
  * and returns result vector.
  * Assumes output vector has been allocated but needs wiped.
  */
real *CSR_matrix_vector_mult(CSRmatrix* csr, real* v, real* r) {
    CSR_matrix_vector(csr, v, r, TRUE, OP_ADD);
    return r;
}

/**
  * CSR_matrix_vector_mult - Helper function does matrix vector multiplication
  * and returns result vector.
  *
  * Allocates result vector for user.
  */
real *CSR_matrix_vector_mult_cr(CSRmatrix* csr, real* v) {

    real* r = (double*) malloc(csr->nrow * sizeof(real));
    CSR_matrix_vector(csr, v, r, TRUE, OP_ADD);
    return r;
}

/* ********************************************* *
 *  Function for matrix - vector multiplication  *
 *  AM init_vec will zero output matrix if set   *
 *  Op is how M[i,j]*v[i] is stored in b         *
 *     op == 1 is addition (M*v)                 *
 *     op == 2 is subtraction                    *
 *     op == 3 is multiplication                 *
 *     op == 4 is division                       *
 *  Any other value won't do anything!           *
 *  All ops work on values in same pos           *
 * AM - Can now use op enum as well e.g. OP_ADD  *
 * ********************************************* */
/*
 * WARNING: the multiplication and division options here don't give the result
 * you might expect (i.e. not the same as doing the matrix multiply into a
 * separate buffer and then explicitly multiplying/dividing) so should probably
 * be avoided.
 */
void CSR_matrix_vector(CSRmatrix *in_CSR, real *v, real *b, int init_vec, int op)
{
    int i, temp = 0;
    real *loc_values;
    int *loc_rowStart, *loc_colIndex, nrow;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_matrix_vector: Aborted. Matrix not valid.\n\n");
        return;
    }

    // Initialize vector *if* requested
    if ( init_vec ) {
        for(i=0; i < in_CSR->nrow; i++)
            b[i] = 0.0;
    }

    // Set local pointers for ease of use
    loc_values   = in_CSR->values;
    loc_rowStart = in_CSR->rowStart;
    loc_colIndex = in_CSR->colIndex;
    nrow = in_CSR->nrow;

    // Iterate over all non-zero values and compute result
    // ----------------------------------------------------------------------------------------
    // -  A few words of how the logic of this function works.                                -
    // -                                                                                      -
    // -  If you set "init_vec" the elements of the result vector are initialized to zero     -
    // -  When (op == 1) and (init_vec==1) then its normal matrix-vector multiplication       -
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

    if ((op == 1) || (op == OP_ADD)) {  // Addition --> normal matrix-vector multiplication
        for(i=0; i < loc_rowStart[nrow]; i++) {
            while ( i == loc_rowStart[temp+1] )
                temp++;
            b[temp] += loc_values[i] * v[loc_colIndex[i]];
        }
    }

    else if ((op == 2) || (op == OP_SUB)) {  // Subtraction --> suitable when you want to perform A*v_1 - B*v_2
        for(i=0; i < loc_rowStart[nrow]; i++) {
            while ( i == loc_rowStart[temp+1] )
                temp++;
            b[temp] -= loc_values[i] * v[loc_colIndex[i]];
        }
    }

    else if ((op == 3) || (op == OP_MULT)) {
        for(i=0; i < loc_rowStart[nrow]; i++) {
            while ( i == loc_rowStart[temp+1] )
                temp++;
            b[temp] *= loc_values[i] * v[loc_colIndex[i]];
        }
    }

    else if ((op == 4) || (op == OP_DIV)) {
        for(i=0; i < loc_rowStart[nrow]; i++) {
            while ( i == loc_rowStart[temp+1] )
                temp++;
            b[temp] /= loc_values[i] * v[loc_colIndex[i]];
        }
    }
}

/* ******************************************* *
 *  Raise a CSR matrix in power of 2 (square)  *
 * ******************************************* */
CSRmatrix *CSR_matrix_square(CSRmatrix *in_CSR)
{
    int i, j, cur_CSR_val, cur_CSC_val;
    real *CSR_values, *CSC_values, cur_sum;
    int *rowStart, *colIndex, mat_rank;
    int *colStart, *rowIndex, count;
    CSCmatrix *converted_matrix;
    CSRmatrix *result;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_matrix_square: Aborted. Matrix not valid.\n\n");
        return(NULL);
    }

    // To be able to raise in square it must be a square matrix
    if ( in_CSR->nrow != in_CSR->ncol ) {
        printf("\nCSR_matrix_square: Aborted. Matrix not valid.\n\n");
        return(NULL);
    }

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_matrix_square: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    if( !CSR_setup(result, in_CSR->nrow, in_CSR->ncol, in_CSR->nzmax*2) ) {
        printf("\nCSR_matrix_square: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Matrix rank (nrow is same as ncol)
    mat_rank = in_CSR->nrow;
    count = 0;

    // Convert CSR matrix to CSC (this will help a lot with the multiplication)
    converted_matrix = convert_CSR_to_CSC(in_CSR);

    // Set a few local variables for convinience and speed
    rowStart = in_CSR->rowStart;
    colIndex = in_CSR->colIndex;
    CSR_values   = in_CSR->values;
    colStart = converted_matrix->colStart;
    rowIndex = converted_matrix->rowIndex;
    CSC_values   = converted_matrix->values;

    for(i=0; i < mat_rank; i++) {
        // Check if sufficient memory for a whole non-zero row is available
        if ( (result->nzmax - result->rowStart[result->nrow]) < result->nrow ) {
            if( !CSR_extend(result) ) {
                printf("\nCSR_matrix_square: Not enough memory for matrix square\n");
                return(NULL);
            }
        }

        // Compute all the elements of the row
        for(j=0; j < mat_rank; j++) {
            cur_sum = 0.0;
            cur_CSR_val = rowStart[i];
            cur_CSC_val = colStart[j];

            // If row is zero no need to do the loop
            if( cur_CSR_val == rowStart[i+1] )
                break;

            // If column is zero then I can continue to the next
            if( cur_CSC_val == colStart[j+1] )
                continue;

            // Scan all pairs
            for(; cur_CSR_val < rowStart[i+1] && cur_CSC_val < colStart[j+1];) {
                // If index match then multiply
                if(colIndex[cur_CSR_val] == rowIndex[cur_CSC_val]) {
                    cur_sum += CSC_values[cur_CSC_val] * CSR_values[cur_CSR_val];
                    cur_CSC_val++;
                    cur_CSR_val++;
                }
                // If one index smaller than other, progress smaller
                else if(colIndex[cur_CSR_val] < rowIndex[cur_CSC_val] )
                    cur_CSR_val++;
                else if(colIndex[cur_CSR_val] > rowIndex[cur_CSC_val])
                    cur_CSC_val++;
            }

            // Add non-zero elements to final sparse matrix
            if( cur_sum != 0.0 ) {
                result->values[count] = cur_sum;
                result->colIndex[count++] = j;

                // Not increament if (i+1) is the last value of rowStart, I do this in the next line.
                if( (i+1) != mat_rank)
                    result->rowStart[i+1]++;
                result->rowStart[mat_rank]++;
            }
        }
        // Add current rowStart to next (only if not the last one)
        if( (i+1) != mat_rank)
            result->rowStart[i+1]+= result->rowStart[i];
    }

    return(result);
}


void CSR_matrix_multiply_reuse(CSRmatrix *in_CSR1, CSRmatrix *in_CSR2, CSRmatrix *out_CSR)
{
    int i, j, k, l;
    real sum;
    static int msglimit = 0;

    /* loop over rows of result matrix (also rows of matrix 1) */
    for (i = 0; i < out_CSR->nrow; i++) {
	/* loop over columns of result matrix (also columns of matrix 2) */
	for (j = out_CSR->rowStart[i]; j < out_CSR->rowStart[i+1]; j++) {
	    int col = out_CSR->colIndex[j];
	    /* need to do dot product of row i of matrix 1 and column col of matrix 2 */
	    sum = 0.0;
	    /* loop over non-zeroes in row i of matrix 1 */
	    for (k = in_CSR1->rowStart[i]; k < in_CSR1->rowStart[i+1]; k++) {
		/* column of this value gives us a row to search in matrix 2 */
		int row = in_CSR1->colIndex[k];
		for (l = in_CSR2->rowStart[row]; l < in_CSR2->rowStart[row+1]; l++) {
		    /* is there a value at this column? */
		    if (in_CSR2->colIndex[l] == col) {
			if (msglimit < 20) {
			    //printf("(%d,%d): adding LOPW(%d,%d) * LOPF(%d,%d)\n", i, col, i, row, row, col);
			    msglimit++;
			}
			sum += in_CSR2->values[l] * in_CSR1->values[k];
			break;
		    }
		}
	    }
	    out_CSR->values[j] = sum;
	}
    }
}

CSRmatrix *CSR_matrix_multiply(CSRmatrix *in_CSR_1, CSRmatrix *in_CSR_2)
{
    CSRmatrix *result = NULL;
    int i, j, k, l, m;
    int idx;
    real sum;

    CSCmatrix *in_CSC_2;

    /* matrix 1 width must equal matrix 2 height */
    if (in_CSR_1->ncol != in_CSR_2->nrow) {
	printf("\nCSR matrix multiply aborted. Incompatible sizes.\n\n");
	return NULL;
    }

    /* allocate result matrix */
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    if (!result) {
	printf("\nCSR matrix multiply aborted. Memory allocation failed.\n\n");
	return NULL;
    }
    if (!CSR_setup(result, in_CSR_1->nrow, in_CSR_2->ncol, in_CSR_1->nzmax + in_CSR_2->nzmax)) {
	printf("\nCSR matrix multiply aborted. Memory allocation failed.\n\n");
	free(result);
	return NULL;
    }

    /* convert matrix 2 to CSC for speed */
    in_CSC_2 = convert_CSR_to_CSC(in_CSR_2);

    /* loop over rows of result matrix (also rows of matrix 1) */
    idx = 0;
    for (i = 0; i < result->nrow; i++) {
	/* loop over columns of result matrix (also cols of CSC matrix 2) */
	for (j = 0; j < result->ncol; j++) {
	    sum = 0.0;
	    /* loop over values in row of matrix 1 */
	    k = in_CSC_2->colStart[j];
	    for (l = in_CSR_1->rowStart[i]; l < in_CSR_1->rowStart[i+1]; l++) {
		while ((in_CSC_2->rowIndex[k] < in_CSR_1->colIndex[l]) &&
		       (k < in_CSC_2->colStart[j+1])) {
		    k++;
		}
		if (k >= in_CSC_2->colStart[j+1]) break;
		if (in_CSC_2->rowIndex[k] == in_CSR_1->colIndex[l]) {
		    sum = sum + in_CSC_2->values[k] * in_CSR_1->values[l];
		}
	    }
	    if (sum != 0.0) {
		/* store sum in result matrix */
		result->colIndex[idx] = j;
		result->values[idx] = sum;
		idx++;
	    }
	}
	/* finalise row in result matrix */
	result->rowStart[i+1] = idx;

	/* check that we still have enough space left */
	if (idx >= (result->nzmax - result->ncol)) {
	    if (!CSR_extend(result)) {
		printf("\nOut of memory extending matrix in CSR multiply.\n\n");
		free(result);
		return NULL;
	    }
	}
    }

    CSC_free(in_CSC_2);
    return result;
}

/**
 * CSR_matrix_add - Add two CSR matrices.
 * 
 * @in_CSR1 First matrix to add
 * @in_CSR2 Second matrix to add
 *
*/
CSRmatrix *CSR_matrix_add(CSRmatrix* in_CSR1, CSRmatrix* in_CSR2) {

    return CSR_matrix_add_sub(in_CSR1, in_CSR2, 0);
}


/**
 * CSR_matrix_sub - Subtract two CSR matrices.
 * 
 * @in_CSR1 First matrix
 * @in_CSR2 Matrix to subtract from in_CSR1
 *
*/
CSRmatrix *CSR_matrix_sub(CSRmatrix* in_CSR1, CSRmatrix* in_CSR2) {

    return CSR_matrix_add_sub(in_CSR1, in_CSR2, 1);
}

/*
 * Add two matrices but reuse an existing structure for the result. Should be much
 * faster than recomputing the non-zero pattern and allocating a new matrix.
 * Non-zero pattern of result must fit within the result matrix passed
 */
void CSR_matrix_add_reuse(CSRmatrix* in_CSR1, CSRmatrix *in_CSR2, CSRmatrix *out_CSR)
{
    int i;
    int ji1, ji2, jo;

    /* loop over rows */
    for (i = 0; i < in_CSR1->nrow; i++) {
	/* loop over populated columns of result matrix */
	ji1 = in_CSR1->rowStart[i];
	ji2 = in_CSR2->rowStart[i];
	for (jo = out_CSR->rowStart[i]; jo < out_CSR->rowStart[i+1]; jo++) {
	    int col = out_CSR->colIndex[jo];

	    /* skip columns of first input matrix that aren't present in result */
	    while ((in_CSR1->colIndex[ji1] < col) && (ji1 < in_CSR1->rowStart[i+1])) {
		/*printf("Warning: skipping column %d of input matrix 1 (%f)\n",
		  in_CSR1->colIndex[ji1], in_CSR1->values[ji1]);*/
		ji1++;
	    }

	    /* skip columns of second input matrix that aren't present in result */
	    while ((in_CSR2->colIndex[ji2] < col) && (ji2 < in_CSR2->rowStart[i+1])) {
		/*printf("Warning: skipping column %d of input matrix 2 (%f)\n",
		  in_CSR2->colIndex[ji2], in_CSR2->values[ji2]);*/
		ji2++;
	    }

	    /* perform the addition */
	    out_CSR->values[jo] = 0.0;
	    if ((in_CSR1->colIndex[ji1] == col) && (ji1 < in_CSR1->rowStart[i+1])) {
		out_CSR->values[jo] += in_CSR1->values[ji1];
		ji1++;
	    }
	    if ((in_CSR2->colIndex[ji2] == col) && (ji2 < in_CSR2->rowStart[i+1])) {
		out_CSR->values[jo] += in_CSR2->values[ji2];
		ji2++;
	    }
	}
    }
}

/* ******************************* *
 *  Add/subtract two CSR matrices  *
 *                                 *
 *  AM Seems add_sub == 0 is add   *
 *     add_sub == 1 is subtract    *
 *   anything else silently fails  *
 * AM - extended to use op enum    *
 * ******************************* */
CSRmatrix *CSR_matrix_add_sub(CSRmatrix *in_CSR_1, CSRmatrix *in_CSR_2, int add_sub)
{
    int i, j, k;
    real *values_1, *values_2;
    int *rowStart_1, *colIndex_1, count_1, count_2, nrow, ncol;
    int *rowStart_2, *colIndex_2, count;
    CSRmatrix *result;

    // Check add_sub option
    if(add_sub != 0 && add_sub != 1 && add_sub != OP_ADD && add_sub != OP_SUB)
        printf("\nCSR_matrix_addition: Aborted. Wrong value of add_sub option.");

    // Check matrices
    if (!CSR_valid_check(in_CSR_1)) {
        printf("\nCSR_matrix_addition: Aborted. First matrix not valid.\n\n");
        return(NULL);
    }
    if (!CSR_valid_check(in_CSR_2)) {
        printf("\nCSR_matrix_addition: Aborted. Second matrix not valid.\n\n");
        return(NULL);
    }

    // To be able to add them their dimensions must agree
    if ( in_CSR_1->nrow != in_CSR_2->nrow ) {
        printf("\nCSR_matrix_addition: Aborted. Matrices have different number of rows.\n\n");
        return(NULL);
    }
    if ( in_CSR_1->ncol != in_CSR_2->ncol ) {
        printf("\nCSR_matrix_addition: Aborted. Matrices have different number of columns.\n\n");
        return(NULL);
    }

    nrow = in_CSR_1->nrow;
    ncol = in_CSR_1->ncol;

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_matrix_addition: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    if( !CSR_setup(result, nrow, ncol, in_CSR_1->nzmax + in_CSR_2->nzmax) ) {
        printf("\nCSR_matrix_addition: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Initialise values
    count_1 = count_2 = count = 0;
    rowStart_1 = in_CSR_1->rowStart;
    colIndex_1 = in_CSR_1->colIndex;
    values_1   = in_CSR_1->values;
    rowStart_2 = in_CSR_2->rowStart;
    colIndex_2 = in_CSR_2->colIndex;
    values_2   = in_CSR_2->values;

    result->rowStart[0] = 0;

    // Go over all rows and add all columns
    for(i=0; i < nrow; i++) {
        // Check if there is enough memory to store a full row.
        // If not reallocate.
        if ( (result->nzmax - result->rowStart[i]) < ncol )
            if( !CSR_extend(result) ) {
                printf("\nCSR_matrix_addition: Not enough memory for matrix addition\n");
                return(NULL);
            }

	if ((add_sub == 0) || (add_sub == OP_ADD)) {
	    j = rowStart_1[i];
	    k = rowStart_2[i];
	    while ((j < rowStart_1[i+1]) && (k < rowStart_2[i+1])) {
		// loop over both matrix rows
		if (colIndex_1[j] < colIndex_2[k]) {
		    // matrix 1 has a value, matrix 2 doesn't
		    result->values[count] = values_1[j];
		    result->colIndex[count++] = colIndex_1[j];
		    j++;
		}
		else if (colIndex_2[k] < colIndex_1[j]) {
		    // matrix 2 has a value, matrix 1 doesn't
		    result->values[count] = values_2[k];
		    result->colIndex[count++] = colIndex_2[k];
		    k++;
		}
		else {
		    // both have a value
		    result->values[count] = values_1[j] + values_2[k];
		    result->colIndex[count++] = colIndex_1[j];
		    j++;
		    k++;
		}
	    }
	    while (j < rowStart_1[i+1]) {
		// remaining values in matrix 1
		result->values[count] = values_1[j];
		result->colIndex[count++] = colIndex_1[j];
		j++;
	    }
	    while (k < rowStart_2[i+1]) {
		// remaining values in matrix 2
		result->values[count] = values_2[k];
		result->colIndex[count++] = colIndex_2[k];
		k++;
	    }
	}
	else if ((add_sub == 1) || (add_sub == OP_SUB)) {
	    j = rowStart_1[i];
	    k = rowStart_2[i];
	    while ((j < rowStart_1[i+1]) && (k < rowStart_2[i+1])) {
		// loop over both matrix rows
		if (colIndex_1[j] < colIndex_2[k]) {
		    // matrix 1 has a value, matrix 2 doesn't
		    result->values[count] = values_1[j];
		    result->colIndex[count++] = colIndex_1[j];
		    j++;
		}
		else if (colIndex_2[k] < colIndex_1[j]) {
		    // matrix 2 has a value, matrix 1 doesn't
		    result->values[count] = -values_2[k];
		    result->colIndex[count++] = colIndex_2[k];
		    k++;
		}
		else {
		    // both have a value
		    result->values[count] = values_1[j] - values_2[k];
		    result->colIndex[count++] = colIndex_1[j];
		    j++;
		    k++;
		}
	    }
	    while (j < rowStart_1[i+1]) {
		// remaining values in matrix 1
		result->values[count] = values_1[j];
		result->colIndex[count++] = colIndex_1[j];
		j++;
	    }
	    while (k < rowStart_2[i+1]) {
		// remaining values in matrix 2
		result->values[count] = -values_2[k];
		result->colIndex[count++] = colIndex_2[k];
		k++;
	    }
	}

        // Set the correct rowStart value for this value
        //result->rowStart[i+1]+= result->rowStart[i];
	result->rowStart[i+1] = count;
    }
    return(result);
}

/**
 * CSR_scalar_mult - In-place scalar by matrix multiplication.
 * @in_CSR Sparse matrix to scale
 * @val Value to scale by
 *
 * Multiplies each value in the matrix by the given value. Computation is done
 * *in-place* i.e. the input matrix is changed.
 * 
 * Modified to return input vector to support chaining.
 */
CSRmatrix* CSR_scalar_mult(CSRmatrix *in_CSR, real val) {

    CSR_scalar_computation(in_CSR, val, OP_MULT);
    return in_CSR;
}

/**
 * CSR_scalar_div - In-place matrix by scalar division.
 * @in_CSR Sparse matrix to divide
 * @val Value to divide by
 *
 * Divides each value in the matrix by the given value. Computation is done
 * *in-place* i.e. the input matrix is changed.
 * 
 * Modified to return input vector to support chaining.
 */
CSRmatrix* CSR_scalar_div(CSRmatrix *in_CSR, real val) {

    CSR_scalar_computation(in_CSR, val, OP_DIV);
    return in_CSR;
}

/* ********************************************************** *
 *  Function to perform a scalar computation to all non-zero  *
 *  values of a CSR matrix.                                   *
 *                                                            *
 *  AM comp == 1 is add                                       *
 *     comp == 2 is subtract                                  *
 *     comp == 3 is multiply                                  *
 *     comp == 4 is divide                                    *
 * ********************************************************** */
void CSR_scalar_computation(CSRmatrix *in_CSR, real val, int comp)
{
    int i;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_scalar_computations: Aborted. Matrix not valid.\n\n");
        return;
    }

    if ((comp == 1) || (comp == OP_ADD)) {
        for(i=0; i < in_CSR->rowStart[in_CSR->nrow]; i++)
            in_CSR->values[i] += val;
    }
    else if ((comp == 2) || (comp == OP_SUB)) {
        for(i=0; i < in_CSR->rowStart[in_CSR->nrow]; i++)
            in_CSR->values[i] -= val;
    }
    else if ((comp == 3) || (comp == OP_MULT)) {
        for(i=0; i < in_CSR->rowStart[in_CSR->nrow]; i++)
            in_CSR->values[i] *= val;
    }
    else if ((comp == 4) || (comp == OP_DIV)) {
        for(i=0; i < in_CSR->rowStart[in_CSR->nrow]; i++)
            in_CSR->values[i] /= val;
    }
}

/*
 * Copies one matrix to another, reusing the structure. The non-zero pattern of the output
 * matrix must be the same as the input, or else the input must be a subset of it
 */
void CSR_copy(CSRmatrix *in_CSR, CSRmatrix *out_CSR)
{
    int i, ji, jo;
    /* loop over rows */
    for (i = 0; i < out_CSR->nrow; i++) {
	/* loop over columns of output matrix */
	ji = in_CSR->rowStart[i];
	for (jo = out_CSR->rowStart[i]; jo < out_CSR->rowStart[i+1]; jo++) {
	    int col = out_CSR->colIndex[jo];

	    /* skip past columns not present in output (there shouldn't be any) */
	    while ((in_CSR->colIndex[ji] < col) && (ji < in_CSR->rowStart[i+1])) {
		printf("Warning: skipping columns in CSR_copy!\n");
		ji++;
	    }

	    if ((in_CSR->colIndex[ji] == col) && (ji < in_CSR->rowStart[i+1])) {
		out_CSR->values[jo] = in_CSR->values[ji];
		ji++;
	    }
	    else {
		out_CSR->values[jo] = 0.0;
	    }
	}
    }
}

/* ***************************************************** *
 *  Create an identical copy of a sparse triplet matrix  *
 * ***************************************************** */
CSRmatrix *CSR_duplicate(CSRmatrix *in_CSR)
{
    CSRmatrix *result;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_duplicate: Aborted. Matrix not valid.\n\n");
        return(NULL);
    }

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_duplicate: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    if( !CSR_setup(result, in_CSR->nrow, in_CSR->ncol, in_CSR->nzmax) ) {
        printf("\nCSR_duplicate: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Copy the memory spaces
    memcpy(result->rowStart, in_CSR->rowStart, sizeof(int) * (in_CSR->nrow+1));
    memcpy(result->colIndex, in_CSR->colIndex, sizeof(int) * in_CSR->nzmax);
    memcpy(result->values,   in_CSR->values, sizeof(real) * in_CSR->nzmax);

    return(result);
}

/**
 * calc_nz_in_toeplitz - Counts number of non-zero values in toeplitz.
 *
 * @col_vec Vector of column values
 * @col_len Length of col_vec
 * @row_vec Vector of row values
 * @row_len Length of row vector.
 */
int calc_nz_in_toeplitz(real* col_vec, int col_len, real *row_vec, int
row_len) {
    // Count non-zero values in input vector.
    // Also compute how many non-zero elements the Sparse
    // matrix will have.

    int i, diag_len, nzmax = 0;
    int max_diag = col_len;
    if (row_len < col_len) {
        max_diag = row_len;
    }

    for(i=0; i < col_len; i++) {
        if(col_vec[i] != 0.0) {
            diag_len = col_len - i;
            if (diag_len > max_diag) {
                nzmax += max_diag;
            } else {
                nzmax += diag_len;
            }
        }
    }

    //Start at 1 as already have main diagonal
    for(i=1; i < row_len; i++) {
        if(row_vec[i] != 0.0) {
            diag_len = row_len - i;
            if (diag_len > max_diag) {
                nzmax += max_diag;
            } else {
                nzmax += diag_len;
            }
        }
    }

    return nzmax;
}

/*
 * CSR_toeplitz - Creates a sparse toeplitz matrix
 * @row_vec Vector of row values
 * @row_len Length of row vector
 * @col_vec Vector of column values
 * @col_len Length of col_vec
 *
 * Constructs a sparse toeplitz matrix using the values from the given row and
 * column. The column overrides any value in the row for the main diagonal.
 *
 * The resultant matrix may be non-square.
 *
 */
CSRmatrix *CSR_toeplitz(real *col_vec, int col_len, real *row_vec, int row_len)
{
    int r, c, nzmax = 0;
    int cur_col = 0, cur_row = 1;
    int no_cols, no_rows;
    CSRmatrix *result;

    /* Confusingly, row_len is number of *cols* and vice-versa! */
    no_cols = row_len;
    no_rows = col_len;

    nzmax = calc_nz_in_toeplitz(col_vec, col_len, row_vec, row_len);

    // Check if there are any non-zero values
    if (nzmax == 0) {
        printf("\nCSR_toeplitz: Aborted. Invalid vector supplied.\n\n");
        return(NULL);
    }

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_toeplitz: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Initialize sparse matrix
    if ( !CSR_setup(result, no_rows, no_cols, nzmax) ) {
        printf("\nCSR_toeplitz: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    result->rowStart[0] = 0;

    // Populate the Toeplitz matrix
    for(r=0; r < no_rows; r++) {
        for(c=0; c < no_cols; c++) {
            if (c > r) {
            	//In top half of matrix, use row vec for vals
                if(row_vec[c-r] != 0.0) {
                    result->values[cur_col] = row_vec[c-r];
                    result->colIndex[cur_col] = c;
                    cur_col++;
                    result->rowStart[cur_row]++;
                }
            } else {
                if (col_vec[r-c] != 0.0) {
                    result->values[cur_col] = col_vec[r-c];
                    result->colIndex[cur_col] = c;
                    cur_col++;
                    result->rowStart[cur_row]++;
                }
            }
        }
        if (cur_row < no_rows) {
            result->rowStart[cur_row+1] = result->rowStart[cur_row];
            cur_row++;
        }
    }

    result->rowStart[result->nrow] = nzmax;

    return(result);
}

/* ******************************************* *
 *  Create a symmetric sparse toeplitz matrix  *
 * ******************************************* */
CSRmatrix *CSR_sym_toeplitz(real *in_vec, int vec_len)
{
    int i, j, k, non_zero_count = 0, nzmax = 0;
    int cur_col = 0, cur_row = 1;
    CSRmatrix *result;

    // Count non-zero values in input vector.
    // Also compute how many non-zero elements the Sparse
    // matrix will have.
    for(i=0; i < vec_len; i++) {
        if(in_vec[i] != 0.0) {
            non_zero_count++;
            if ( i == 0 ) {
                //First value is central diagonal (hence only one)
                nzmax += vec_len;
            } else {
                //Every other value has two diagonals of decreasing length
                nzmax += (2 * (vec_len-i));
            }
        }
    }

    // Check if there are any non-zero values
    if (nzmax == 0 || non_zero_count == 0) {
        printf("\nCSR_toeplitz: Aborted. Invalid vector supplied.\n\n");
        return(NULL);
    }

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_toeplitz: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Initialize sparse matrix
    if ( !CSR_setup(result, vec_len, vec_len, nzmax) ) {
        printf("\nCSR_toeplitz: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Populate the Toeplitz matrix
    for(i=0; i < vec_len; i++) {
        k = i;
        for(j=0; j < vec_len; j++) {
            if(in_vec[k] != 0.0) {
                result->values[cur_col] = in_vec[k];
                result->colIndex[cur_col++] = j;
                result->rowStart[cur_row]++;
            }
            if (i > j) k--;
            else       k++;
        }
        if (cur_row < vec_len) {
            result->rowStart[cur_row+1] = result->rowStart[cur_row];
            cur_row++;
        }
    }

    result->rowStart[result->nrow] = nzmax;

    return(result);
}


/* **************************************************************** *
 *  Kronecker product of an identity matrix as first argument and   *
 *  a sparse matrix as second. The resulting matrix is also sparse  *
 * **************************************************************** */
CSRmatrix *CSR_kron_eye_mat(CSRmatrix *in_CSR, int eye_rank)
{
    CSRmatrix *result;
    int i, j, new_nrow, new_ncol, new_nzmax, mem_count, local_ncol;
    int start, temp;
    real *dest, *orig;
    int *int_dest, *int_orig;
    int nnz;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_kron_eye_mat: Aborted. Matrix not valid.\n\n");
        return(NULL);
    }

    nnz = in_CSR->rowStart[in_CSR->nrow];

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_kron_eye_mat: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Calculate the number of rows, columns and non-zero elements
    // of the new sparse matrix
    new_nrow  = in_CSR->nrow * eye_rank;
    new_ncol  = in_CSR->ncol * eye_rank;
    new_nzmax = in_CSR->nzmax * eye_rank;

    // Initialize sparse matrix
    if ( !CSR_setup(result, new_nrow, new_ncol, new_nzmax) ) {
        printf("\nCSR_kron_eye_mat: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Replicate the values as many times as needed
    dest = result->values;
    orig = in_CSR->values;
    mem_count = nnz;
    for(i=0; i < eye_rank; i++) {
        memcpy(&dest[i*mem_count], orig, mem_count*sizeof(real));
    }

    // Copy the column indexes
    int_dest = result->colIndex;
    int_orig = in_CSR->colIndex;
    local_ncol = in_CSR->ncol;
    for(i=0; i < eye_rank; i++) {
        start = i * nnz;
        temp = i * local_ncol;
        for(j=0; j < nnz; j++) {
            int_dest[start+j] = int_orig[j] + temp;
        }
    }

    // Copy the row starts
    int_dest = result->rowStart;
    int_orig = in_CSR->rowStart;
    for(i=0; i < eye_rank; i++) {
        start = i * in_CSR->nrow;
        temp = i * nnz;
        for(j=0; j < in_CSR->nrow; j++) {
            int_dest[start+j] = int_orig[j] + temp;
        }
    }

    // Set the number of non-zero elements
    result->rowStart[new_nrow] = nnz * eye_rank;

    return(result);
}

/*
 * Computes the kronecker product of any two sparse matrices
 */
CSRmatrix *CSR_kronecker_product(CSRmatrix *a, CSRmatrix *b)
{
    CSRmatrix *result;
    int new_nrow, new_ncol, new_nzmax;
    int i, j, k, l, idx, row;
    real aval;
    int acol;

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    
    new_nrow = a->nrow * b->nrow;
    new_ncol = a->ncol * b->ncol;
    new_nzmax = a->rowStart[a->nrow] * b->rowStart[b->nrow];
    if (!CSR_setup(result, new_nrow, new_ncol, new_nzmax)) {
	printf("\nCSR_kronecker_product: Aborted. Allocation failed.\n\n");
	free(result);
	return NULL;
    }

    idx = 0;
    row = 0;
    result->rowStart[row] = 0;
    /* loop over rows of A */
    for (i = 0; i < a->nrow; i++) {
	/* loop over rows of B */
	for (j = 0; j < b->nrow; j++) {
	    /* loop over columns of A */
	    for (k = a->rowStart[i]; k < a->rowStart[i+1]; k++) {
		/* get the value to multiply B by */
		aval = a->values[k];
		acol = a->colIndex[k];
		/* loop over columns of B */
		for (l = b->rowStart[j]; l < b->rowStart[j+1]; l++) {
		    result->values[idx] = b->values[l] * aval;
		    result->colIndex[idx] = (acol * b->ncol) + b->colIndex[l];
		    idx++;
		}
	    }

	    result->rowStart[row+1] = idx;
	    row++;
	}	
    }

    return result;
}

/* **************************************************************** *
 *  Kronecker product of an identity matrix as second argument and  *
 *  a sparse matrix as first. The resulting matrix is also sparse   *
 * **************************************************************** */
CSRmatrix *CSR_kron_mat_eye(CSRmatrix *in_CSR, int eye_rank)
{
    CSRmatrix *result;
    int new_nrow, new_ncol, new_nzmax;
    int i, j, k, idx, row;

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
  
    new_nrow = in_CSR->nrow * eye_rank;
    new_ncol = in_CSR->ncol * eye_rank;
    new_nzmax = in_CSR->nzmax * eye_rank;

    if (!CSR_setup(result, new_nrow, new_ncol, new_nzmax)) {
	printf("\nCSR_kron_mat_eye: Aborted. Allocation failed.\n\n");
	free(result);
	return NULL;
    }

    idx = 0;
    row = 0;
    result->rowStart[row] = 0;
    /* loop over rows of original matrix */
    for (i = 0; i < in_CSR->nrow; i++) {
	/* loop over rows of identity matrix */
	for (j = 0; j < eye_rank; j++) {
	    /* loop over columns of original matrix */
	    for (k = in_CSR->rowStart[i]; k < in_CSR->rowStart[i+1]; k++) {
		result->values[idx] = in_CSR->values[k];
		result->colIndex[idx] = (in_CSR->colIndex[k] * eye_rank) + j;
		idx++;
	    }
	    result->rowStart[row+1] = idx;
	    row++;
	}
    }

    return result;
}

/* **************************************************************** *
 *  Kronecker product of a toeplitz matrix and                      *
 *  identity matrix. The resulting matrix is sparse.                *
 *                                                                  *
 *  AM: first 2 args represent symmetric toeplitz matrix; top row   *
 *  and length. 3 arg is length of side of identity matrix.         *
 *  The result of kron product of toeplitz and identity             *
 *  is another (sparser) toeplitz matrix.                           *
 * **************************************************************** */
CSRmatrix *CSR_kron_toeplitz_eye(real *in_vec, int vec_len, int eye_rank)
{
    real *toep_vec;
    int toep_vec_len, i;

    // Allocate memory for the vector needed by the Toeplitz function
    toep_vec_len = vec_len * eye_rank;
    toep_vec = (real *)calloc(toep_vec_len, sizeof(real));

    // Construct the resultant Toeplitz vector
    for(i=0; i < vec_len; i++) {
        if ( in_vec[i] != 0.0 )
            toep_vec[i*eye_rank] = in_vec[i];
    }

    // Create and return the Toeplitz sparse matrix
    return(CSR_sym_toeplitz(toep_vec, toep_vec_len));
}


/* ************************************************ *
 *  Function that creates square identity matrices  *
 * ************************************************ */
CSRmatrix *CSR_create_eye(int eye_rank)
{
    CSRmatrix *result;
    int i;

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_create_eye: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Initialize sparse matrix
    if ( !CSR_setup(result, eye_rank, eye_rank, eye_rank) ) {
        printf("\nCSR_create_eye: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Set values of struct
    for(i=0; i < eye_rank; i++) {
        result->values[i] = 1.0;
        result->colIndex[i] = i;
        result->rowStart[i] = i;
    }

    // Set value of non-zero elements
    result->rowStart[eye_rank] = eye_rank;

    return(result);
}


/* ***************************************************** *
 *  Function to free allocated memory for sparse matrix. *
 * ***************************************************** */
int CSR_free(CSRmatrix *in_CSR)
{
    free(in_CSR->values);
    free(in_CSR->colIndex);
    free(in_CSR->rowStart);
    free(in_CSR);

    return 1;
}

int CSC_free(CSCmatrix *in_CSC)
{
    free(in_CSC->values);
    free(in_CSC->rowIndex);
    free(in_CSC->colStart);
    free(in_CSC);

    return 1;
}

/* ******************************************************* *
 *  Internal function to check if sparse matrix is valid.  *
 * ******************************************************* */
int CSR_valid_check(CSRmatrix *in_CSR)
{

    // Check if memory is allocated
    if ( in_CSR->values == NULL || in_CSR->colIndex == NULL || in_CSR->rowStart == NULL ) {
	printf("One of arrays is null\n");
        return(0);
    }

    // Check if valid row and column values
    if ( in_CSR->nrow <= 0 || in_CSR->ncol <= 0 ) {
	printf("nrow or ncol is negative or zero\n");
        return(0);
    }

    // Check if non-zero elements exist
    if (in_CSR->rowStart[in_CSR->nrow] == 0) {
	printf("nnz is zero\n");
        return(0);
    }

    if (in_CSR->nzmax < in_CSR->rowStart[in_CSR->nrow]) {
	printf("nnz > nzmax\n");
	return(0);
    }

    // All OK
    return(1);
}

/* ****************************************************************** *
 *  Internal function to extend the memory of a CSR matrix.           *
 *  The implementation adds an additional 50% to the previous         *
 *  allocation. Maximum allocation check is performed (full matrix).  *
 * ****************************************************************** */
int CSR_extend(CSRmatrix *in_CSR)
{
    int max_alloc, new_alloc, old_alloc, i;
    old_alloc = in_CSR->nzmax;

    // Maximum allocation => Full matrix
    max_alloc = in_CSR->nrow * in_CSR->ncol;

    // Add 50% more space
    new_alloc = (int)ceil(in_CSR->nzmax * 1.5);

    // If new allocation more than full then set to full
    if ( new_alloc > max_alloc )
        new_alloc = max_alloc;

    // Check that at least a full row is added, anything less is not worth it really
    if ( (new_alloc - in_CSR->nzmax) < in_CSR->nrow ) {
        new_alloc += in_CSR->nrow;
    }

    // Re-allocate
    in_CSR->values = (real *)realloc(in_CSR->values, new_alloc * sizeof(real));
    in_CSR->colIndex = (int *)realloc(in_CSR->colIndex, new_alloc * sizeof(int));

    if (DEBUG) {
        //Initialise the new memory to dummy values
        //Should never be read until set anyway though
        for (i = old_alloc; i < new_alloc; i++) {
            in_CSR->values[i] = 0.0;
            in_CSR->colIndex[i] = -1;
        }
    }

    if ( in_CSR->values == NULL || in_CSR->colIndex == NULL ) {
        printf("\nCSR_extend: Memory (re)allocation failed.\n\n");
        return(0);
    }

    // Set new nzmax
    in_CSR->nzmax = new_alloc;

    // All OK
    return(1);
}


/* ************************************** *
 *  Output the sparse matrix as an image  *
 * ************************************** */
void CSR_PGM_output(CSRmatrix *in_CSR, char *filename)
{
    int count = 0;
    int row, col;
    FILE *ofp;
    char buf[2], final_filename[100];

    // Check if filename passed as argument.
    // If yes then open the stream and write everything there.
    if ( filename == NULL ) {
        printf("\nCSR_PGM_output: Filename not given.\n\n");
        return;
    }
    else {
        sprintf(final_filename, "%s.pgm", filename);
        ofp = fopen(final_filename, "wb");  
        if ( ofp == NULL ) {
            printf("\nCSR_PGM_output: Unable to open file.\n\n");
            return;
        }
    }

    // P5 is gray scale binary
    fprintf(ofp, "P5\n%d %d\n%d\n", in_CSR->nrow, in_CSR->ncol, 255);

    // Go over all elements and write pixels
    for(row=0; row < in_CSR->nrow; row++) {
        for(col=0; col < in_CSR->ncol; col++) {
            if( (in_CSR->colIndex[count] == col) && (count < in_CSR->rowStart[row+1]) ) {
                sprintf(buf, "%c", 0);
                count++;
            }
            else {
                sprintf(buf, "%c", 255);
            }
            fwrite(buf, 1, 1, ofp);
        }
    }

    // Close image file
    fclose(ofp);
}


/*
 * Selectively remove rows from a CSR matrix. rowsToKeep is an array of
 * indices of rows that should NOT be removed, len is the length of this
 * array. The array should be in ascending order. If the array contains
 * indices that are larger than the matrix, they will be ignored.
 */
int CSR_cut_rows(CSRmatrix *csr, int *rowsToKeep, int len)
{
    int newnrow;
    int newnnz;
    int i, j;
    real *newvals;
    int *newcolidx, *newrowstart;

    /* remove indices that are larger than the matrix */
    i = len;
    while ((rowsToKeep[i - 1] >= csr->nrow) && (i > 1)) i--;
    len = i;
    if (len == 0) {
	/* no rows left! */
	csr->nrow = 0;
	return 1;
    }
    newnrow = len;

    /* count up total number of entries that will remain, and initialise new rowStart */
    newrowstart = (int *)malloc((len+1) * sizeof(int));
    if (!newrowstart) return 0;
    newnnz = 0;
    newrowstart[0] = 0;
    for (i = 0; i < len; i++) {
	newnnz += (csr->rowStart[rowsToKeep[i]+1] - csr->rowStart[rowsToKeep[i]]);
	newrowstart[i+1] = newnnz;
    }

    /* allocate space for values and indices */
    newvals = (real *)malloc(newnnz * sizeof(real));
    newcolidx = (int *)malloc(newnnz * sizeof(int));
    if ((!newvals) || (!newcolidx)) {
	free(newrowstart);
	return 0;
    }

    /* populate new values and indices */
    /* loop over new rows */
    for (i = 0; i < len; i++) {
	int rowlen = newrowstart[i+1] - newrowstart[i];
	for (j = 0; j < rowlen; j++) {
	    newvals[newrowstart[i] + j] = csr->values[csr->rowStart[rowsToKeep[i]] + j];
	    newcolidx[newrowstart[i] + j] = csr->colIndex[csr->rowStart[rowsToKeep[i]] + j];
	}
    }

    /* replace data in matrix */
    csr->nrow = newnrow;
    csr->nzmax = newnnz;
    free(csr->colIndex);
    csr->colIndex = newcolidx;
    free(csr->rowStart);
    csr->rowStart = newrowstart;
    free(csr->values);
    csr->values = newvals;

    return 1;
}

/*
 * Selectively remove columns from a CSR matrix. colsToKeep is an array of
 * indices of columns that should NOT be removed, len is the length of this
 * array. The array should be in ascending order. If the array contains
 * indices that are larger than the matrix, they will be ignored.
 */
int CSR_cut_cols(CSRmatrix *csr, int *colsToKeep, int len)
{
    int newncol;
    int i, j, k;
    real *newvals;
    int *newcolidx, *newrowstart;
    int idx;

    /* remove indices that are larger than the matrix */
    i = len;
    while ((colsToKeep[i - 1] >= csr->ncol) && (i > 1)) i--;
    len = i;
    if (len == 0) {
	/* no columns left! */
	csr->ncol = 0;
	for (i = 0; i < csr->nrow+1; i++) {
	    csr->rowStart[i] = 0;
	}
	return 1;
    }
    newncol = len;

    /* allocate space - worst case, we need as much as before */
    newrowstart = (int *)malloc((csr->nrow+1) * sizeof(int));
    newvals = (real *)malloc(csr->nzmax * sizeof(real));
    newcolidx = (int *)malloc(csr->nzmax * sizeof(int));
    if ((!newrowstart) || (!newvals) || (!newcolidx)) return 0;
    
    idx = 0;
    /* loop over rows */
    for (i = 0; i < csr->nrow; i++) {
	newrowstart[i] = idx;
	k = 0;
	/* loop over old columns */
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    /* see if we're keeping this column */
	    for (; k < len; k++) {
		if (colsToKeep[k] == csr->colIndex[j]) {
		    /* yes, keeping this one */
		    newcolidx[idx] = k;
		    newvals[idx] = csr->values[j];
		    idx++;
		    break;
		}
		/* it's not in colsToKeep, skip it */
		if (colsToKeep[k] > csr->colIndex[j]) break;
	    }
	}
    }
    newrowstart[i] = idx;

    /* replace data in matrix */
    csr->ncol = newncol;
    free(csr->rowStart);
    csr->rowStart = newrowstart;
    free(csr->colIndex);
    csr->colIndex = newcolidx;
    free(csr->values);
    csr->values = newvals;

    return 1;
}

/*
 * Selectively zero rows in a matrix
 */
void CSR_zero_rows(CSRmatrix *csr, int *rowsToKeep, int len)
{
    int i, j, k;
    int keep;

    /* loop over rows of matrix */
    for (i = 0; i < csr->nrow; i++) {
	keep = 0;

	/* are we keeping this one? */
	for (k = 0; k < len; k++) {
	    if (rowsToKeep[k] == i) {
		keep = 1;
		break;
	    }
	}

	if (!keep) {
	    /* zero it */
	    for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
		csr->values[j] = 0.0;
		//if (csr->colIndex[j] == i) csr->values[j] = 1.0;
	    }
	}
    }
}

/*
 * Selectively zero columns in a matrix
 */
void CSR_zero_cols(CSRmatrix *csr, int *colsToKeep, int len)
{
    int i, j, k;
    int *c2k;

    /* convert colsToKeep into a boolean array */
    c2k = (int *)malloc(csr->ncol * sizeof(int));
    memset(c2k, 0, csr->ncol * sizeof(int));
    for (i = 0; i < len; i++) {
	c2k[colsToKeep[i]] = 1;
    }

    /* loop over rows of matrix */
    for (i = 0; i < csr->nrow; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    if (!c2k[csr->colIndex[j]]) {
		csr->values[j] = 0.0;
		//if (csr->colIndex[j] == i) csr->values[j] = 1.0;
	    }
	}
    }

    free(c2k);
}


/*
 * Scales the rows of the matrix by the given vector, as if multiplying it by matrix with the
 * vector as its diagonal
 */
void CSR_diagonal_scale(CSRmatrix *csr, double *vec)
{
    int i, j;
    for (i = 0; i < csr->nrow; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    csr->values[j] *= vec[i];
	}
    }
}

/*
 * Scales the columns of the matrix by the given vector, as if multiplying a matrix with
 * the vector as its diagonal by this matrix
 */
void CSR_column_scale(CSRmatrix *csr, double *vec)
{
    int i, j;
    for (i = 0; i < csr->nrow; i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    csr->values[j] *= vec[csr->colIndex[j]];
	}
    }
}


CSRmatrix *CSR_get_sub_matrix(CSRmatrix *csr, int row, int col, int height, int width)
{
    CSRmatrix *result;
    int nnz;
    int i, j;
    int idx;

    if ((row + height) > csr->nrow) {
	printf("CSR_get_sub_matrix failed: %d rows required, matrix only has %d\n", row + height,
	       csr->nrow);
	return NULL;
    }
    if ((col + width) > csr->ncol) {
	printf("CSR_get_sub_matrix failed: %d cols required, matrix only has %d\n", col + width,
	       csr->ncol);
	return NULL;
    }

    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    /* calculate nnz for new matrix */
    nnz = 0;    
    for (i = row; i < (row+height); i++) {
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    if ((csr->colIndex[j] >= col) && (csr->colIndex[j] < (col+width))) {
		nnz++;
	    }
	}
    }

    /* allocate space */
    CSR_setup(result, height, width, nnz);

    /* extract rows and columns */
    idx = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < height; i++) {
	for (j = csr->rowStart[i+row]; j < csr->rowStart[i+row+1]; j++) {
	    if ((csr->colIndex[j] >= col) && (csr->colIndex[j] < (col+width))) {
		result->colIndex[idx] = csr->colIndex[j] - col;
		result->values[idx] = csr->values[j];
		idx++;
	    }
	}
	result->rowStart[i+1] = idx;
    }
    
    return result;
}


#define MINIMUM_VALUE 1e-40

/*
 * Saves a matrix to a PETSc format binary data file. Values less than 1e-12 are removed as
 * they're probably artefacts of our optimisations and shouldn't really be there at all.
 *
 * This is a good format for doing comparisons with Matlab as it's very simple
 * and there also exists a Matlab writer for it.
 *
 * This won't work on big endian machines, or when real is defined as float.
 */
int CSR_save_petsc(char *filename, CSRmatrix *csr)
{
    FILE *f;
    unsigned char header[0x10];
    unsigned char *buf;
    unsigned char *dbl;
    int i, j;
    int nnz;
    int check;

    f = fopen(filename, "wb");
    if (!f) {
	printf("Error opening matrix file %s for writing\n", filename);
	return 0;
    }

    /* first generate the header */
    header[0] = 0x00; /* magic number for sparse matrix */
    header[1] = 0x12;
    header[2] = 0x7b;
    header[3] = 0x50;

    header[4] = (csr->nrow >> 24) & 0xff;
    header[5] = (csr->nrow >> 16) & 0xff;
    header[6] = (csr->nrow >> 8)  & 0xff;
    header[7] = (csr->nrow >> 0)  & 0xff;

    header[8] = (csr->ncol >> 24) & 0xff;
    header[9] = (csr->ncol >> 16) & 0xff;
    header[10] = (csr->ncol >> 8) & 0xff;
    header[11] = (csr->ncol >> 0) & 0xff;

    /* count number of (actual) non-zeroes */
    nnz = 0;
    for (i = 0; i < csr->rowStart[csr->nrow]; i++) {
	if (fabs(csr->values[i]) > MINIMUM_VALUE) {
	    nnz++;
	}
    }

    header[12] = (nnz >> 24) & 0xff;
    header[13] = (nnz >> 16) & 0xff;
    header[14] = (nnz >> 8)  & 0xff;
    header[15] = (nnz >> 0)  & 0xff;

    fwrite(header, 1, 0x10, f);

    /* next come the row lengths */
    /* allocate a buffer for reversing the data */
    if (nnz > csr->nrow) {
	buf = malloc(nnz * sizeof(double));
    }
    else {
	buf = malloc(csr->nrow * sizeof(double));
    }
    if (!buf) {
	printf("Out of memory while saving matrix to %s\n", filename);
	fclose(f);
	return 0;
    }

    check = 0;
    for (i = 0; i < csr->nrow; i++) {
	int rowlen = 0;

	/* count the actual length of this row */
	for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
	    if (fabs(csr->values[j]) > MINIMUM_VALUE) {
		rowlen++;
	    }
	}

	/* convert to big endian */
	buf[i*4  ] = (rowlen >> 24) & 0xff;
	buf[i*4+1] = (rowlen >> 16) & 0xff;
	buf[i*4+2] = (rowlen >> 8)  & 0xff;
	buf[i*4+3] = (rowlen >> 0)  & 0xff;

	check += rowlen;
    }
    fwrite(buf, 1, csr->nrow * 4, f);

    if (check != nnz) {
	printf("nnz doesn't match sum of row lengths!!\n");
    }

    /* now column indices */
    j = 0;
    for (i = 0; i < csr->rowStart[csr->nrow]; i++) {
	if (fabs(csr->values[i]) > MINIMUM_VALUE) {
	    buf[j*4  ] = (csr->colIndex[i] >> 24) & 0xff;
	    buf[j*4+1] = (csr->colIndex[i] >> 16) & 0xff;
	    buf[j*4+2] = (csr->colIndex[i] >> 8)  & 0xff;
	    buf[j*4+3] = (csr->colIndex[i] >> 0)  & 0xff;
	    j++;
	}
    }
    fwrite(buf, 1, nnz * 4, f);

    if (j != nnz) {
	printf("nnz not reached after column index processing!\n");
    }

    /* finally actual values */
    j = 0;
    for (i = 0; i < csr->rowStart[csr->nrow]; i++) {
	if (fabs(csr->values[i]) > MINIMUM_VALUE) {
	    dbl = (unsigned char *)&csr->values[i];
	    buf[j*8  ] = dbl[7];
	    buf[j*8+1] = dbl[6];
	    buf[j*8+2] = dbl[5];
	    buf[j*8+3] = dbl[4];
	    buf[j*8+4] = dbl[3];
	    buf[j*8+5] = dbl[2];
	    buf[j*8+6] = dbl[1];
	    buf[j*8+7] = dbl[0];
	    j++;
	}
    }
    fwrite(buf, 1, nnz * 8, f);

    if (j != nnz) {
	printf("nnz not reached after value processing!\n");
    }

    free(buf);
    fclose(f);

    return 1;
}

/*
 * Loads a matrix from a PETSc format binary data file
 */
CSRmatrix *CSR_load_petsc(char *filename)
{
    CSRmatrix *result, *trans;
    int nrow, ncol, nnz;
    FILE *f;
    unsigned char header[0x10];
    unsigned char *buf1, *buf2, *buf3;
    unsigned char doublebuf[8];
    int i, j, k, idx, cidx;
    int colsize, row;

    f = fopen(filename, "rb");
    if (!f) {
	printf("Error opening matrix file %s\n", filename);
	return NULL;
    }

    /* first is a 16 byte header */
    fread(header, 1, 0x10, f);

    /* data is all big endian */
    ncol = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | (header[7]);
    nrow = (header[8] << 24) | (header[9] << 16) | (header[10] << 8) | (header[11]);
    nnz = (header[12] << 24) | (header[13] << 16) | (header[14] << 8) | (header[15]);

    /* allocate the matrix */
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    if (!result) {
	fclose(f);
	return NULL;
    }
    CSR_setup(result, nrow, ncol, nnz);

    /* allocate temporary storage - for column sizes... */
    buf1 = (unsigned char *)malloc(ncol * 4);
    /* ... for row indices... */
    buf2 = (unsigned char *)malloc(nnz * 4);
    /* ... for actual data */
    buf3 = (unsigned char *)malloc(nnz * 8);

    /* read actual data from file */
    fread(buf1, 1, ncol*4, f);
    fread(buf2, 1, nnz*4, f);
    fread(buf3, 1, nnz*8, f);
    fclose(f);

    /*
     * FIXME: this could be made a lot simpler - I thought the format was CSC so wrote
     * all this conversion code, but it's actually CSR. For now I've just added a
     * transpose on the end to correct it.
     */

    /* now convert to CSR */
    /* loop over rows of result */
    cidx = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < nrow; i++) {
	/* loop over columns of input */
	idx = 0;
	for (j = 0; j < ncol; j++) {
	    /* loop over values within the column */
	    colsize = (buf1[(j<<2)] << 24) | (buf1[(j<<2)+1] << 16) |
		(buf1[(j<<2)+2] << 8) | (buf1[(j<<2)+3]);
	    for (k = 0; k < colsize; k++) {
		/* is this one in our row? */
		row = (buf2[(idx << 2)] << 24) | (buf2[(idx << 2) + 1] << 16) |
		    (buf2[(idx << 2) + 2] << 8) | (buf2[(idx << 2) + 3]);
		if (row == i) {
		    /* found next entry - add it to CSR */
		    result->colIndex[cidx] = j;

		    /* this won't work on big endian systems... */
		    doublebuf[7] = buf3[(idx << 3)];
		    doublebuf[6] = buf3[(idx << 3) + 1];
		    doublebuf[5] = buf3[(idx << 3) + 2];
		    doublebuf[4] = buf3[(idx << 3) + 3];
		    doublebuf[3] = buf3[(idx << 3) + 4];
		    doublebuf[2] = buf3[(idx << 3) + 5];
		    doublebuf[1] = buf3[(idx << 3) + 6];
		    doublebuf[0] = buf3[(idx << 3) + 7];

		    result->values[cidx] = *((double *)&doublebuf[0]);

		    cidx++;
		}

		idx++;
	    }
	}
	result->rowStart[i+1] = cidx;
    }

    free(buf1);
    free(buf2);
    free(buf3);

    trans = CSR_transpose(result);
    CSR_free(result);

    return trans;
}


/* =================== * 
 *    CSC functions    * 
 * =================== */


/* ***************************************** *
 *  Function to allocate memory for the CSC  *
 *  matrix.                                  *
 *                                           *
 *  csc   : the matrix to be setup            *
 *  r     : number of rows                    *
 *  c     : number of columns                 *
 *  nzmax : max number of non-zero entries    *
 * ***************************************** */
int CSC_setup(CSCmatrix *csc, int r, int c, int nzmax)
{
    int max_nzmax;

    csc->nrow=r;
    csc->ncol=c;
    csc->nzmax=nzmax;

    // Check if requested non_zero elements are more than the maximum (full matrix)
    max_nzmax = r * c;
    if ( max_nzmax < nzmax )
        csc->nzmax = max_nzmax;
    else
        csc->nzmax = nzmax;

    csc->rowIndex = (int *)calloc(nzmax, sizeof(int));
    csc->colStart = (int *)calloc(c+1, sizeof(int));
    csc->values   = (real *)calloc(nzmax, sizeof(real));

    // Check if all memory was allocated successfully
    if ( csc->values == NULL || csc->rowIndex == NULL || csc->colStart == NULL ) {
        printf("\nCSC matrix memory allocation failed.\n");

        // Free any memory allocated
        if (csc->values != NULL) free(csc->values);
        if (csc->rowIndex != NULL) free(csc->rowIndex);
        if (csc->colStart != NULL) free(csc->colStart);

        return(0);
    }

    // All OK
    return(1);
}

/**
 * CSR_transpose - Transposes a CSR matrix.
 * @in_CSR CSR matrix to transpose
 *
 * Based on code to create CSC. Should return a new matrix that is the
 * transpose of its input.
 */
CSRmatrix *CSR_transpose(CSRmatrix *in_CSR)
{
    CSRmatrix *new_CSR;
    int count, i, j, k;

    //Allocate 
    new_CSR = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( new_CSR == NULL ) {
        printf("\nCSR_transpose: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Initialize instance - note size of rows and columns swapped
    if( !CSR_setup(new_CSR, in_CSR->ncol, in_CSR->nrow, in_CSR->nzmax) ) {
        printf("\nCSR_transpose: Aborted. Allocation failed.\n\n");
        free(new_CSR);
        return(NULL);
    }

    count = 0;
    for(i=0; i < new_CSR->nrow; i++) {
        for(j=0; j < new_CSR->ncol; j++) {
            for(k=in_CSR->rowStart[j]; k < in_CSR->rowStart[j+1]; k++) {

                if ( in_CSR->colIndex[k] > i )
                    break;
                if(in_CSR->colIndex[k] == i) {
                    new_CSR->values[count] = in_CSR->values[k];
                    new_CSR->colIndex[count++] = j;
                    new_CSR->rowStart[i+1]++;
                    break;
                }
            }
        }
        new_CSR->rowStart[i+1] += new_CSR->rowStart[i];
    }
    new_CSR->rowStart[new_CSR->nrow] = count;

    return(new_CSR);
}

/*
 * Compares the two matrices for equality within the specified tolerance
 *
 * Returns 0 if they match, 1 if not
 */
int CSR_compare(CSRmatrix *csr1, CSRmatrix *csr2, double tolerance)
{
    int i, j, k;

    /* check they're the same size */
    if ((csr1->nrow != csr2->nrow) || (csr1->ncol != csr2->ncol)) {
	return 1;
    }

    /* loop over rows of both matrices */
    for (i = 0; i < csr1->nrow; i++) {
	k = csr2->rowStart[i];
	for (j = csr1->rowStart[i]; j < csr1->rowStart[i+1]; j++) {
	    while ((csr2->colIndex[k] < csr1->colIndex[j]) && (k < csr2->rowStart[i+1])) {
		/* csr2 has an entry that csr1 doesn't */
		/* it has to be less than tolerance */
		if (fabs(csr2->values[k]) > tolerance) {
		    printf("Matrix compare failed, row %d, col %d, csr2 has extra value (1)\n",
			   i, csr2->colIndex[k]);
		    return 1;
		}
		k++;
	    }

	    if ((k >= csr2->rowStart[i+1]) || (csr1->colIndex[j] < csr2->colIndex[k])) {
		/* csr1 has an entry that csr2 doesn't */
		/* it has to be less than tolerance */
		if (fabs(csr1->values[j]) > tolerance) {
		    printf("Matrix compare failed, row %d, col %d, csr1 has extra value\n",
			   i, csr1->colIndex[j]);
		    return 1;
		}
	    }
	    else if (csr1->colIndex[j] == csr2->colIndex[k]) {
		/* both have an entry */
		/* it has to match to within tolerance */
		if (fabs(csr1->values[j] - csr2->values[k]) > tolerance) {
		    printf("Matrix compare failed, row %d, col %d, matrices differ\n", i,
			   csr1->colIndex[j]);
		    return 1;
		}
		k++;
	    }
	}
	while (k < csr2->rowStart[i+1]) {
	    /* csr2 has an entry that csr1 doesn't */
	    /* it has to be less than tolerance */
	    if (fabs(csr2->values[k]) > tolerance) {
		printf("Matrix compare failed, row %d, col %d, csr2 has extra value\n",
		       i, csr2->colIndex[k]);
		return 1;
	    }
	    k++;
	}
    }

    return 0;
}


/**
 * For sparse matrix A, vector x, calculates vector b = A*x.
 *
 * It is assumed x and b are allocated. All values of b are set by this
 * function.
 *
 * @param A Sparse N x M matrix
 * @param x Vector of length M
 * @param b Vector of length N
 */
void CSR_fast_mat_vec_mult(CSRmatrix* A, double* x, double* b) {
    int irow, jcol;
    double sum;
    for (irow=0; irow<A->nrow; irow++) {
        sum = 0.0;
        for (jcol=A->rowStart[irow]; jcol<A->rowStart[irow+1]; jcol++) {
            sum += A->values[jcol]*x[A->colIndex[jcol]];
        }
        b[irow] = sum;
    }
}

/**
 * Create a sparse block diagonal matrix
 *
 * Creates a new matrix consisting of the two matrices combined diagonally,
 * i.e. ( A 0 )
 *      ( 0 B )
 *
 * A and B dims don't have to match but they must both be square
 *
 * @param A First matrix
 * @param B Second matrix
 * @return The block diagonal matrix
 */
CSRmatrix* CSR_blk_diag(CSRmatrix *A, CSRmatrix* B)
{

    if (!CSR_valid_check(A)) {
        printf("\nCSR_blk_diag: Aborted. Matrix not valid.\n\n");
        return (NULL);
    }
    if (!CSR_valid_check(B)) {
        printf("\nCSR_blk_diag: Aborted. Matrix not valid.\n\n");
        return (NULL);
    }

    if (A->nrow != A->ncol) {
        printf("\nCSR_blk_diag: Aborted. A is not a square matrix.\n\n");
        return (NULL);
    }
    if (B->nrow != B->ncol) {
        printf("\nCSR_blk_diag: Aborted. B is not a square matrix.\n\n");
        return (NULL);
    }

    int nnzA = A->rowStart[A->nrow];
    int nnzB = B->rowStart[B->nrow];
    int newnnz = nnzA+nnzB;

    CSRmatrix* result;

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_blk_diag: Aborted. Memory allocation failed.\n");
        return (NULL);
    }

    if( !CSR_setup(result, A->nrow+B->nrow, A->ncol+B->ncol, newnnz) ) {
        printf("\nCSR_blk_diag: Aborted. Allocation failed.\n\n");
        free(result);
        return (NULL);
    }

    // Copy the memory spaces, first A ...
    memcpy(result->values,   A->values, sizeof(real) * nnzA);
    memcpy(result->rowStart, A->rowStart, sizeof(int) * (A->nrow+1));
    memcpy(result->colIndex, A->colIndex, sizeof(int) * nnzA);


    // ... then B
    memcpy(&result->values[nnzA],   B->values, sizeof(real) * nnzB);
    memcpy(&result->rowStart[A->nrow+1], &B->rowStart[1], sizeof(int) * (B->nrow));
    memcpy(&result->colIndex[nnzA], B->colIndex, sizeof(int) * nnzB);

    // Adjust rowStart and colIndex for B data
    int i;
    for (i = nnzA; i<newnnz; i++) result->colIndex[i] += A->ncol;
    for (i = A->nrow+1; i<A->nrow+B->nrow+1; i++) result->rowStart[i] +=nnzA;

    return result;
}

/**
 * Kronecker product of A (x) B where A is actually a diagonal matrix
 *
 * @param in_CSR Matrix B
 * @param diag The diagonal values of A
 * @param diag_size The size of diag
 * @return The mat
 */
CSRmatrix *CSR_kron_diag_mat(CSRmatrix *in_CSR, double* diag, int diag_size)
{
    CSRmatrix *result;
    int i, j, new_nrow, new_ncol, new_nzmax, mem_count, local_ncol;
    int start, temp;
    real *dest, *orig;
    real *orig_scaled;

    int *int_dest, *int_orig;
    int nnz;

    if (!CSR_valid_check(in_CSR)) {
        printf("\nCSR_kron_diag_mat: Aborted. Matrix not valid.\n\n");
        return(NULL);
    }

    nnz = in_CSR->rowStart[in_CSR->nrow];

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_kron_diag_mat: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Calculate the number of rows, columns and non-zero elements
    // of the new sparse matrix
    new_nrow  = in_CSR->nrow * diag_size;
    new_ncol  = in_CSR->ncol * diag_size;
    new_nzmax = 0;
    for (i=0; i<diag_size; i++) {
        if (diag[i] != 0.0) new_nzmax += in_CSR->nzmax;
    }
    // Initialize sparse matrix
    if ( !CSR_setup(result, new_nrow, new_ncol, new_nzmax) ) {
        printf("\nCSR_kron_diag_mat: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    // Replicate the values as many times as needed
    dest = result->values;
    orig = in_CSR->values;
    mem_count = nnz;
    orig_scaled = (real *)malloc(sizeof(real)*mem_count);
    int count = 0;
    for(i=0; i < diag_size; i++) {
        if (diag[i] != 0.0) {
            if (diag[i] != 1.0) {
                memcpy(&orig_scaled[0], orig, mem_count*sizeof(real));
                for(j=0; j < mem_count; j++) {
                    orig_scaled[j] *= diag[i];
                }
                memcpy(&dest[count*mem_count], orig_scaled, mem_count*sizeof(real));
            } else {
                memcpy(&dest[count*mem_count], orig, mem_count*sizeof(real));
            }
            count++;
        }
    }
    free(orig_scaled);

    // Copy the column indexes
    int_dest = result->colIndex;
    int_orig = in_CSR->colIndex;
    local_ncol = in_CSR->ncol;
    start = 0;
    //temp = 0;
    for(i=0; i < diag_size; i++) {
//        start = i * nnz;
       temp = i * local_ncol;
        if (diag[i] != 0.0) {

            for(j=0; j < nnz; j++) {
                int_dest[start+j] = int_orig[j] + temp;
            }
         //   temp+=local_ncol;
            start+=nnz;
        }
    }

    // Copy the row starts
    int_dest = result->rowStart;
    int_orig = in_CSR->rowStart;
    temp = 0;
    start = 0;
    for(i=0; i < diag_size; i++) {
        start = i * in_CSR->ncol;
        //temp = i * nnz;
        if (diag[i] != 0.0) {

            for(j=0; j < in_CSR->nrow; j++) {
                int_dest[start+j] = int_orig[j] + temp;
            }
            temp+=nnz;
        } else {
            //if (i>0) tmp = int_dest[start-1];
            for(j=0; j < in_CSR->nrow; j++) {
                int_dest[start+j] = temp;
            }
        }
    }

    // Set the number of non-zero elements
    result->rowStart[new_nrow] = nnz * count;

    return(result);
}


/**
 * Multiplies two sparse matrices which are square and only populated on the diagonal
 *
 * @param in_CSR_1 First matrix
 * @param in_CSR_2 Second matrix
 * @return The resultant matrix C = AB
 */
CSRmatrix *CSR_matrix_mult_sqr_diag(CSRmatrix* in_CSR_1, CSRmatrix* in_CSR_2)
{
    CSRmatrix *result;
    int i;

    // Check input matrices
    /* must be square and matrix 1 dims must equal matrix 2 dims */
    if (in_CSR_1->ncol != in_CSR_2->nrow || in_CSR_1->nrow != in_CSR_2->ncol) {
        printf("\nCSR_matrix_mult_sqr_diag aborted. Incompatible sizes.\n\n");
        return NULL;
    }

    /* allocate result matrix */
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));
    if (!result) {
        printf("\nCSR_matrix_mult_sqr_diag aborted. Memory allocation failed.\n\n");
        return NULL;
    }
    // if (!CSR_setup(result, in_CSR_1->nrow, in_CSR_2->ncol, in_CSR_1->nrow)) {
    if (!CSR_setup(result, in_CSR_1->nrow, in_CSR_2->ncol, in_CSR_1->nrow+in_CSR_2->rowStart[in_CSR_2->nrow])) {
        printf("\nCSR_matrix_mult_sqr_diag aborted. Memory allocation failed.\n\n");
        free(result);
        return NULL;
    }

    // Set values of struct
    int idx = 0;
    int irow, j;
    for(irow=0; irow < in_CSR_1->nrow; irow++) {
        result->rowStart[irow] = idx;
        for (i = in_CSR_1->rowStart[irow]; i < in_CSR_1->rowStart[irow+1]; i++) {
            for (j = in_CSR_2->rowStart[irow]; j < in_CSR_2->rowStart[irow+1]; j++) {
                result->values[idx] = in_CSR_1->values[i]*in_CSR_2->values[j];
                result->colIndex[idx] = in_CSR_2->colIndex[j];
                idx++;
            }

        }

    }
    // Set value of non-zero elements
    result->rowStart[in_CSR_1->nrow] = idx;

    return(result);
}

/**
 * Selectively zero rows in a sparse matrix
 *
 * The main difference between this and CSR_zero_rows is that this one creates a new
 * matrix with *only* non-zero values, rather than using the old matrix and
 * setting values to zero, which can have performance implications. The
 * input array rowsToKeep also differs: it has either a zero or a one for each
 * row, a one indicates that particular row's values are to be kept, rather than
 * a list containing just the indices of the rows to keep
 *
 * @param csr Input matrix
 * @param rowsToKeep Array of size N rows, 0 for discard, 1 to keep
 * @return The new matrix, same rank as csr but with the indicated rows zeroed.
 */
CSRmatrix* CSR_zero_rows_new(CSRmatrix *csr, int *rowsToKeep)
{
    int i, j; //, k;
    //int keep;

    CSRmatrix *result;

    // Allocate struct instance
    result = (CSRmatrix *)malloc(sizeof(CSRmatrix));

    // Check memory allocation
    if ( result == NULL ) {
        printf("\nCSR_create_eye: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Find out how many non-zeroes in new matrix
    int nnz = 0;
    for (i=0; i<csr->nrow;i++) {
        if (rowsToKeep[i] == 1) {
            nnz+= csr->rowStart[i+1]-csr->rowStart[i];
        }
    }

    // Initialize sparse matrix
    if ( !CSR_setup(result, csr->nrow, csr->ncol, nnz) ) {
        printf("\nCSR_create_eye: Aborted. Allocation failed.\n\n");
        free(result);
        return(NULL);
    }

    /* loop over rows of matrix */
    int nnzr = 0;
    nnz = 0;
    result->rowStart[0] = 0;
    for (i = 0; i < csr->nrow; i++) {
        nnzr = 0;
        if (rowsToKeep[i] == 1) { // Keeping this row
            for (j = csr->rowStart[i]; j < csr->rowStart[i+1]; j++) {
                result->colIndex[nnz] = csr->colIndex[j];
                result->values[nnz]   = csr->values[j];
                nnzr++;
                nnz++;
            }
        }
        result->rowStart[i+1] = result->rowStart[i]+nnzr;
    }

    return result;
}


/* ************************************************* *
 *  Convert a sparse matrix in CSR form to CSC form  *
 * ************************************************* */
CSCmatrix *convert_CSR_to_CSC(CSRmatrix *in_CSR)
{
    int i, j, k, count;
    CSCmatrix *new_CSC;

    int *rowpos;

    // Allocate CSC instance
    new_CSC = (CSCmatrix *)malloc(sizeof(CSCmatrix));

    // Check memory allocation
    if ( new_CSC == NULL ) {
        printf("\nconvert_CSR_to_CSC: Aborted. Memory allocation failed.\n");
        return(NULL);
    }

    // Initialize instance
    if( !CSC_setup(new_CSC, in_CSR->nrow, in_CSR->ncol, in_CSR->nzmax) ) {
        printf("\nconvert_CSR_to_CSC: Aborted. Allocation failed.\n\n");
        free(new_CSC);
        return(NULL);
    }

    rowpos = (int *)malloc(sizeof(int) * in_CSR->nrow);
    memset(rowpos, 0, sizeof(int) * in_CSR->nrow);

    //
    count = 0;
    for(i=0; i < new_CSC->ncol; i++) {
        for(j=0; j < new_CSC->nrow; j++) {
	    k = rowpos[j] + in_CSR->rowStart[j];
	    if (k < in_CSR->rowStart[j+1]) {
		if (in_CSR->colIndex[k] == i) {
                    new_CSC->values[count] = in_CSR->values[k];
                    new_CSC->rowIndex[count++] = j;
                    new_CSC->colStart[i+1]++;

		    rowpos[j]++;
		}
	    }
        }
        new_CSC->colStart[i+1] += new_CSC->colStart[i];
    }

    free(rowpos);

    return(new_CSC);
}

/* ****************************************** *
 *  Print out a CSC matrix as a full matrix.  *
 * ****************************************** */
void CSC_print(CSCmatrix *csc, char *filename)
{
    int count = 0;
    int row, col;
    FILE *stream;

    // Check if filename passed as argument.
    // If yes then open the stream and write everything there.
    // If no then output to standard output.
    if ( filename == NULL )
        stream = stdout;
    else {
        stream = fopen(filename, "w");
        if ( stream == NULL ) {
            printf("\nCSC_print: Unable to open file. Function will print in stdout.\n\n");
            stream = stdout;
        }
    }

    fprintf(stream, "\n\nCSC Library: Printing sparse matrix");
    fprintf(stream, "\n(Warning: The matrix is printed transposed)");
    fprintf(stream, "\n-------------------------------------------\n\n");

    for(col=0; col < csc->ncol; col++) {
        for(row=0; row < csc->nrow; row++) {
            if( (csc->rowIndex[count] == row) && (count < csc->colStart[col+1]) ) {
                fprintf(stream, "%9.4lf  ", csc->values[count]);
                count++;
            }
            else {
                //printf("%9.4lf  ", 0.0);
                fprintf(stream, "%9d  ", 0);
            }
        }
        fprintf(stream, "\n");
    }

    // If file stream opened then close it
    if ( stream != stdout )
        fclose(stream);
}

