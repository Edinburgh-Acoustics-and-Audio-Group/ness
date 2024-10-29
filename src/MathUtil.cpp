/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014-2015. All rights reserved.
 */

#include "MathUtil.h"

#include <cmath>
#include <cstring>
using namespace std;

#ifndef BRASS_ONLY

#define EPSILON 2.2204460492503131e-16

static double sign(double val)
{
    if (val < 0.0) return -1.0;
    else if (val > 0.0) return 1.0;
    else return 0.0;
}

double newtonSolver(double a, double b, double M, double K, double alpha, double offset, double one_sided,
		    double *phi_ra)
{
    double coeff = K / (alpha + 1.0);
    double r = 1.0;
    double R, F, temp;
    double phi_ra2;

    for (int nn = 0; nn < 5; nn++) {
	double ra = r + a;
	double rae = fabs(ra) - offset;
	double ae = fabs(a) - offset;

	double sra = sign(ra);
	double srae = sign(rae);
	double sae = sign(ae);
	double sa = sign(a);
	double sr = sign(r);

	phi_ra2 = 0.5 * coeff * (1.0 - 0.5 * one_sided * (1.0 - sra)) * (1.0 + srae)
	    * pow(fabs(rae), alpha + 1.0);
	double phi_a = 0.5 * coeff * (1.0 - 0.5 * one_sided * (1.0 - sa)) * (1.0 + sae)
	    * pow(fabs(ae), alpha + 1.0);
	double phi_prime = 0.5 * K * sra * (1.0 - 0.5 * one_sided * (1.0 - sra))
	    * (1.0 + srae) * pow(fabs(rae), alpha);
	if (fabs(r) > EPSILON) {
	    R = sr * (phi_ra2 - phi_a) / fabs(r);
	    F = r + M*R + b;
	    temp = (r * phi_prime - phi_ra2 + phi_a) / (r*r);
	}
	else {
	    R = 0.5 * (K * (1.0 - 0.5 * one_sided * (1.0 - sae))) * (1.0 + sae)
		* pow(fabs(ae), alpha);
	    F = r + M*R + b;
	    temp = 0.5 * (0.5 * alpha * (alpha+1.0) * K
			  * (1.0 - 0.5 * one_sided * (1.0 - sae)) * (1.0 + sae)
			  * pow(fabs(ae), (alpha - 1.0)));
	}
	r = r - F / (1.0 + M * temp);
	//printf("%f %f %f %f %f %f %f %f %f %f\n", ra, rae, ae, phi_ra2, phi_a, phi_prime,
	//       R, F, temp, r);
    }
    if (phi_ra) *phi_ra = phi_ra2;
    return -R;
}

/*
 * Create and initialise a Newton solver structure. The matrix M must have the same
 * non-zero structure as the matrix that is later passed to newtonSolver (with no diagonal),
 * but it doesn't have to contain the same values.
 */
newton_solver_t *newtonSolverCreate(int max_it, double tol, int jacobi_it, double jacobi_tol,
				    CSRmatrix *M)
{
    int len = M->nrow;
    newton_solver_t *newton = new newton_solver_t;
    if (!newton) return NULL;

    newton->maxIterations = max_it;
    newton->jacobiIterations = jacobi_it;
    newton->tolerance = tol;
    newton->jacobiTolerance = jacobi_tol;

    newton->IMQ = CSR_duplicate(M);
    newton->J = CSR_duplicate(M);

    newton->coeff = new double[len];
    newton->phi_a = new double[len];
    newton->fac2 = new double[len];
    newton->fac3 = new double[len];
    newton->temp = new double[len];
    newton->F = new double[len];
    newton->IMQdiag = new double[len];
    newton->x = new double[len];
    newton->d = new double[len];
    newton->xp = new double[len];
    newton->jacobiRows = new int[len];

    return newton;
}

void newtonSolverFree(newton_solver_t *newton)
{
    CSR_free(newton->IMQ);
    CSR_free(newton->J);
    delete(newton->coeff);
    delete(newton->phi_a);
    delete(newton->fac2);
    delete(newton->fac3);
    delete(newton->temp);
    delete(newton->F);
    delete(newton->IMQdiag);
    delete(newton->x);
    delete(newton->d);
    delete(newton->xp);
    delete(newton->jacobiRows);
    delete(newton);
}


/*
 * All parameters are inputs except:
 *  - phi_ra and R are outputs
 *  - r is both an input and an output
 *  - newton structure contains all the temporary vectors required by Newton
 *
 * The M matrix should not contain any diagonal entries; the diagonal should be in the Mdiag vector instead.
 */
void newtonSolverVector(newton_solver_t *newton, double *r, double *a, double *b, CSRmatrix *M, double *Mdiag,
			double *q, double *K, double *alpha, double *phi_ra, double *R)
{
    int len = M->nrow;
    int i, j, nn;
    double val;
    int ci;

    int k;
    double resid, rtol = newton->jacobiTolerance * newton->jacobiTolerance;
    int row, hasnz;
    double Jdiag;
    double Fval;
    int numjacobirows;

    /* compute phi_a, fac2 and fac3 */
    for (i = 0; i < len; i++) {
	newton->coeff[i] = K[i] / (alpha[i] + 1.0);

	if (a[i] > 0) {
	    double pow_a_alpham1 = pow(a[i], alpha[i]-1.0);
	    double pow_a_alpha = pow_a_alpham1 * a[i];
	    double pow_a_alphap1 = pow_a_alpha * a[i];

	    newton->phi_a[i] = newton->coeff[i] * pow_a_alphap1;
	    newton->fac2[i] = 0.5 * alpha[i] * (alpha[i]+1.0) * K[i] * pow_a_alpham1;
	    newton->fac3[i] = K[i] * pow_a_alpha;
	}
	else {
	    newton->phi_a[i] = 0.0;
	    newton->fac2[i] = 0.0;
	    newton->fac3[i] = 0.0;
	}
    }

    /*
     * compute IMQ (= I + M*Q)
     * Like M, the diagonal is stored separately
     */
    for (i = 0; i < len; i++) {
	// compute diagonal entry, including adding identity matrix
	newton->IMQdiag[i] = (Mdiag[i] * q[i]) + 1.0;

	// compute off-diagonal entries
	for (j = newton->IMQ->rowStart[i]; j < newton->IMQ->rowStart[i+1]; j++) {
	    newton->IMQ->values[j] = M->values[j] * q[newton->IMQ->colIndex[j]];
	}
    }

    /* Main Newton iteration loop */
    for (nn = 0; nn < newton->maxIterations; nn++) {

	/* compute phi_ra, R and temp */
	for (i = 0; i < len; i++) {
	    double phi_prime = 0.0, phi_diff;
	    double ra = r[i] + a[i];
	    phi_ra[i] = 0.0;
	    if (ra > 0.0) {
		double pow_ra_alpha = pow(ra, alpha[i]);
		double pow_ra_alphap1 = pow_ra_alpha * ra;
		phi_ra[i] = newton->coeff[i] * pow_ra_alphap1;
		phi_prime = K[i] * pow_ra_alpha;
	    }
	    phi_diff = phi_ra[i] - newton->phi_a[i];

	    if ((r[i] > EPSILON) || (r[i] < -EPSILON)) {
		R[i] = (phi_diff / r[i]);
		newton->temp[i] = ((r[i] * phi_prime) - phi_diff) / (r[i]*r[i]);
	    }
	    else {
		R[i] = newton->fac3[i];
		newton->temp[i] = newton->fac2[i];
	    }	    
	}

	/*
	 * This loop sets up for the Jacobi solver, performing several different operations in one go:
	 *   - computes RHS for Jacobi solver (F = IMQ*r + M*R + b)
	 *   - computes system matrix for Jacobi solver (J = I + M*(temp+q))
	 *   - computes initial residual, if small enough don't need to enter Jacobi loop at all
	 *   - computes inverse of diagonal for Jacobi solver
	 *   - makes a list of rows with non-zero off-diagonal entries for the Jacobi solver
	 * we take advantage of the fact that J, M and IMQ have identical structure...
	 */
        numjacobirows = 0;
	resid = 0.0;
	for (i = 0; i < len; i++) {
	    /* compute diagonal part of F[i] */
	    val = (newton->IMQdiag[i] * r[i]) + (Mdiag[i] * R[i]);

	    /* compute J's diagonal */
	    Jdiag = (Mdiag[i] * (newton->temp[i] + q[i])) + 1.0;

	    /* invert J's diagonal for Jacobi */
	    newton->d[i] = 1.0 / Jdiag;
	    hasnz = 0;

	    /* now handle off-diagonals */
	    for (j = M->rowStart[i]; j < M->rowStart[i+1]; j++) {
		/* non-diagonal part of F[i] */
		ci = M->colIndex[j];
		val += (M->values[j] * R[ci]) + (newton->IMQ->values[j] * r[ci]);

		/* compute J's non-diagonal values */
		newton->J->values[j] = M->values[j] * (newton->temp[ci] + q[ci]);

		/* mark this row for Jacobi */
		if (newton->J->values[j] != 0.0) hasnz = 1;
	    }

	    /* compute final value of F[i] */
	    Fval = val + b[i];

	    /* update dot product of RHS (initial residual) */
	    resid += Fval * Fval;

	    /* initialise this row for Jacobi */
	    if (hasnz) {
		/* row has non-zero off-diagonal entries, add it to the list */
		newton->jacobiRows[numjacobirows] = i;
		numjacobirows++;
		newton->x[i] = 0.0;
	    }
	    else {
		/* row only has diagonal. solve it right now */
		newton->x[i] = Fval * newton->d[i];
		newton->xp[i] = Fval * newton->d[i];
	    }

	    /* store F[i] */
	    newton->F[i] = Fval;
	}
	    
	/* actual Jacobi loop to solve Jx = F */
	k = 0;
	while ((k < newton->jacobiIterations) && (resid > rtol)) {
	    //printf("Jacobi iteration %d, residual %.20f\n", k, r);
	    
	    /* xp = Dinv(b - Rx) */
	    for (i = 0; i < numjacobirows; i++) {
		row = newton->jacobiRows[i];
		val = newton->F[row];
		for (j = newton->J->rowStart[row]; j < newton->J->rowStart[row+1]; j++) {
		    val -= newton->J->values[j] * newton->x[newton->J->colIndex[j]];
		}
		val *= newton->d[row];
		newton->xp[row] = val;
	    }
	    
	    /* do a second iteration to get back from xp to x, and compute the relative residual */
	    resid = 0.0;
	    for (i = 0; i < numjacobirows; i++) {
		row = newton->jacobiRows[i];
		val = newton->F[row];
		for (j = newton->J->rowStart[row]; j < newton->J->rowStart[row+1]; j++) {
		    val -= newton->J->values[j] * newton->xp[newton->J->colIndex[j]];
		}
		val *= newton->d[row];
		newton->x[row] = val;
		
		/* compute relative residual */
		val -= newton->xp[row];
		resid += val*val;
	    }
	    
	    k += 2;
	}

	/* update r from solver result */
	val = 0.0;
	for (i = 0; i < len; i++) {
	    r[i] = r[i] - newton->x[i];
	    val += newton->x[i] * newton->x[i];
	}

	if (val < (newton->tolerance*newton->tolerance)) break;
    }

    /* update R: R = R + Q*r */
    for (i = 0; i < len; i++) {
	R[i] += q[i] * r[i];
    }
}

#endif

void interp1(double *x, double *v, double *xq, int lenx, int lenxq, double *result)
{
    int i, j;
    for (i = 0; i < lenxq; i++) {
	if (xq[i] < x[0]) {
	    // far LHS
	    result[i] = v[0];
	}
	else if (xq[i] >= x[lenx - 1]) {
	    // far RHS
	    result[i] = v[lenx - 1];
	}
	else {
	    // find the points it falls between
	    for (j = 0; j < (lenx-1); j++) {
		if ((x[j] <= xq[i]) && (x[j+1] > xq[i])) {
		    // interpolate from those points
		    double alpha = (xq[i] - x[j]) / (x[j+1] - x[j]);
		    result[i] = (1.0 - alpha) * v[j] + alpha * v[j+1];
		}
	    }
	}
    }
}

/*
 * As above, but x and v are interleaved in a single array
 */
void interp1_interleaved(double *xandv, double *xq, int lenx, int lenxq, double *result)
{
    int i, j;
    for (i = 0; i < lenxq; i++) {
	if (xq[i] < xandv[0]) {
	    // far LHS
	    result[i] = xandv[1];
	}
	else if (xq[i] >= xandv[(lenx - 1)*2]) {
	    // far RHS
	    result[i] = xandv[(lenx - 1)*2 + 1];
	}
	else {
	    // find the points it falls between
	    for (j = 0; j < (lenx-1); j++) {
		if ((xandv[j*2] <= xq[i]) && (xandv[(j+1)*2] > xq[i])) {
		    // interpolate from those points
		    double alpha = (xq[i] - xandv[j*2]) / (xandv[(j+1)*2] - xandv[j*2]);
		    result[i] = (1.0 - alpha) * xandv[j*2+1] + alpha * xandv[(j+1)*2+1];
		}
	    }
	}
    }
}

double factorial(double n)
{
    double i;
    double result = 1.0;
    for (i = 1.0; i <= n; i += 1.0) {
	result *= i;
    }
    return result;
}


bool croutDecomposition(double *A, double *L, double *U, int n)
{
    int i, j, k;
    double sum;

    for (i = 0; i < n; i++) {
	U[(i*n)+i] = 1.0;
    }

    for (j = 0; j < n; j++) {
	for (i = j; i < n; i++) {
	    sum = 0.0;
	    for (k = 0; k < j; k++) {
		sum = sum + L[(i*n)+k] * U[(k*n)+j];

	    }
	    L[(i*n)+j] = A[(i*n)+j] - sum;
	}
	for (i = j; i < n; i++) {
	    sum = 0.0;
	    for (k = 0; k < j; k++) {
		sum = sum + L[(j*n)+k] * U[(k*n)+i];
	    }
	    if (L[(j*n)+j] == 0.0) return false;
	    U[(j*n)+i] = (A[(j*n)+i] - sum) / L[(j*n)+j];
	}
    }
    return true;
}

void croutSolve(double *L, double *U, double *b, double *x, double *y, int n)
{
    int i, j;
    double sum;

    // forward solve
    for (i = 0; i < n; i++) {
	sum = 0.0;
	for (j = 0; j < i; j++) {
	    sum += L[(i*n)+j] * y[j];
	}
	y[i] = (b[i] - sum) / L[(i*n)+i];
    }

    // backward solve
    for (i = (n-1); i >= 0; i--) {
	sum = 0.0;
	for (j = (i+1); j < n; j++) {
	    sum += U[(i*n)+j] * x[j];
	}
	x[i] = (y[i] - sum); // U has unit diagonal so no need to divide
    }
}

/*
 * Gets the eigenvalues (in val) and the corresponding eigenvectors (in vec)
 * of a dense symmetric matrix A, size NxN, using the Jacobi method.
 * The eigenvalues are not sorted by this function.
 */
void getEigenvalues(int N, double *A, double *val, double *vec)
{
    int i, j, m, n;
    double threshold, theta, tau, sum;
    double p, q, r, s, t;
    double *tmp1, *tmp2;

    /* allocate temporary storage */
    tmp1 = new double[N];
    tmp2 = new double[N];

    /* initialise eigenvectors to identity matrix */
    for (m = 0; m < N; m++) {
	for (n = 0; n < N; n++) {
 	    if (m == n) vec[(m*N)+n] = 1.0;
	    else vec[(m*N)+n] = 0.0;
	}
    }
    for (m = 0; m < N; m++) {
	/* initialise to diagonal of matrix */
	tmp1[m] = A[(m*N)+m];
	val[m] = A[(m*N)+m];

	tmp2[m] = 0.0;
    }

    /* do Jacobi iterations */
    for (i = 0; i < 50; i++) {
	/* sum off-diagonal elements */
	sum = 0.0;
	for (m = 0; m < (N-1); m++) {
	    for (n = (m+1); n < N; n++) {
		sum += fabs(A[(m*N)+n]);
	    }
	}
	if (sum == 0.0) break; /* done! */

	/* set threshold */
	if (i < 5) threshold = 0.2*sum/((double)(N*N));
	else threshold = 0.0;

	for (m = 0; m < (N-1); m++) {
	    for (n = (m+1); n < N; n++) {
		if ((i >= 4) && (fabs(A[(m*N)+n]) == 0.0)) {
		    A[(m*N)+n] = 0.0;
		}
		else if (fabs(A[(m*N)+n]) > threshold) {
		    r = val[n] - val[m];
		    if (fabs(A[(m*N)+n]) == 0.0) {
			t = A[m*N+n] / r;
		    }
		    else {
			theta = 0.5*r/(A[m*N+n]);
			t = 1.0 / (fabs(theta) + sqrt(1.0+theta*theta));
			if (theta < 0.0) t = -t;
		    }
		    p = 1.0 / sqrt(1.0+t*t);
		    s = t*p;
		    tau = s / (1.0+p);
		    r = t * A[m*N+n];
		    tmp2[m] -= r;
		    val[m] -= r;
		    tmp2[n] += r;
		    val[n] += r;
		    A[m*N+n] = 0.0;

		    /* perform rotations */
		    for (j = 0; j <= (m-1); j++) {
			q=A[j*N+m];
			r=A[j*N+n];
			A[j*N+m]=q-s*(r+q*tau);
			A[j*N+n]=r+s*(q-r*tau);
		    }
		    for (j = m+1; j <= (n-1); j++) {
			q=A[m*N+j];
			r=A[j*N+n];
			A[m*N+j]=q-s*(r+q*tau);
			A[j*N+n]=r+s*(q-r*tau);
		    }
		    for (j = n+1; j < N; j++) {
			q=A[m*N+j];
			r=A[n*N+j];
			A[m*N+j]=q-s*(r+q*tau);
			A[n*N+j]=r+s*(q-r*tau);
		    }
		    for (j = 0; j < N; j++) {
			q=vec[j*N+m];
			r=vec[j*N+n];
			vec[j*N+m]=q-s*(r+q*tau);
			vec[j*N+n]=r+s*(q-r*tau);
		    }
		}
	    }
	}

	for (m = 0; m < N; m++) {
	    tmp1[m] += tmp2[m];
	    val[m] = tmp1[m];
	    tmp2[m] = 0.0;
	}
    }

    /* free temporary storage */
    delete[] tmp1;
    delete[] tmp2;
}

void denseMatrixVectorMultiply(double *A, double *x, double *b, int m, int n)
{
    int i, j;
    double val;

    /* loop over rows of matrix, and elements of result vector */
    for (i = 0; i < m; i++) {
	val = 0.0;
	/* loop over columns of matrix, and elements of source vector */
	for (j = 0; j < n; j++) {
	    val += x[j] * A[(j*m)+i];
	}
	b[i] = val;
    }
}

void denseMatrixVectorMultiplyTransposed(double *A, double *x, double *b, int m, int n)
{
    int i, j;
    double val;

    /* loop over rows of transposed matrix (columns of original), and result vector */
    for (i = 0; i < n; i++) {
	val = 0.0;
	/* loop over columns of transposed matrix (rows of original), and source vector */
	for (j = 0; j < m; j++) {
	    val += x[j] * A[(i*m)+j];
	}
	b[i] = val;
    }
}

void denseCholeskyDecomp(int N, double *B, double *L, double *U)
{
    int i, j, k;
    double sum;

    memcpy(L, B, N*N*sizeof(double));

    for (i = 0; i < N; i++) {
	for (j = i; j < N; j++) {
	    sum = L[(i*N)+j];
	    for (k = (i-1); k >= 0; k--) {
		sum -= L[(i*N)+k] * L[(j*N)+k];
	    }
	    if (i == j) {
		if (sum <= 0.0) {
		    // matrix is not positive definite
		    return;
		}
		L[(i*N)+i] = sqrt(sum);
		U[(i*N)+i] = sqrt(sum);
	    }
	    else {
		L[(j*N)+i] = sum / L[(i*N)+i];
		U[(i*N)+j] = sum / L[(i*N)+i];
		
		/* make sure other elements are zeroed */
		L[(i*N)+j] = 0.0;
		U[(j*N)+i] = 0.0;
	    }
	}
    }
}

void invertLowerTriangle(int N, double *L, double *I)
{
    int i, j, k;
    double sum;

    memcpy(I, L, N*N*sizeof(double));

    for (i = 0; i < N; i++) {
	I[(i*N)+i] = 1.0 / L[(i*N)+i];
	for (j = i+1; j < N; j++) {
	    sum = 0.0;
	    for (k = i; k < j; k++) {
		sum -= I[(j*N)+k] * I[(k*N)+i];
	    }
	    I[(j*N)+i] = sum / L[(j*N)+j];
	}
    }
}

void transposeDenseMatrix(int N, double *in, double *out)
{
    int i, j;
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    out[(j*N)+i] = in[(i*N)+j];
	}
    }
}

void denseMatrixMatrixMultiply(int N, double *A, double *B, double *out)
{
    /* out(i,j) = row i in A * column j in B */
    int i, j, k;
    double sum;
    for (i = 0; i < N; i++) {
	for (j = 0; j < N; j++) {
	    sum = 0.0;
	    for (k = 0; k < N; k++) {
		sum += A[(i*N)+k] * B[(k*N)+j];
	    }
	    out[(i*N)+j] = sum;
	}
    }
}
