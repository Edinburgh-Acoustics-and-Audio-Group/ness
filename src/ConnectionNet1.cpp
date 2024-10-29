/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2016. All rights reserved.
 *
 * Author: James Perry (j.perry@epcc.ed.ac.uk)
 */

#include "ConnectionNet1.h"
#include "Logger.h"
#include "GlobalSettings.h"
#include "TaskWholeConnection.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

/*struct TaylorSeries {
    TaylorSeries() {
	valid = false;
    }

    void initialise(double a, double p) {
	int i;

	this->a = a;
	valid = true;

	double val = pow(a, p);
	double inva = 1.0 / a;
	double denom = 1.0;

	c[0] = val;
	for (i = 1; i < 8; i++) {
	    // derive
	    val = (val * inva) * p;
	    p -= 1.0;
	    c[i] = val / denom;
	    denom = denom * (double)(i+1);
	}
    }

    double calculate(double x) {
	double x2 = x*x;
	double x3 = x2*x;
	double x4 = x2*x2;
	double x5 = x4*x;
	double x6 = x4*x2;
	double x7 = x4*x3;
	return c[0] + c[1]*x + c[2]*x2 + c[3]*x3 +
	    c[4]*x4 + c[5]*x5 + c[6]*x6 + c[7]*x7;
    }

    bool valid;
    double a;
    double c[8];
    };*/

static inline double fastPow2(double a, double b) {
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
  return u.d;
}

// should be much more precise with large b
static inline double fastPow3(double a, double b) {
  // calculate approximation with fraction of the exponent
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
 
  // exponentiation by squaring with the exponent's integer part
  // double r = u.d makes everything much slower, not sure why
  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }
 
  return r * u.d;
}

/*static inline double fastPow(double a, double b) {
    if (a == 0.0) return 0.0;
    return pow(a, b);
    }*/

#define fastPow pow

/*static inline double fastPow(double a, double b) {
    return a*a*a;
    }*/

#define EPS 2.220446049250313e-16

ConnectionNet1::ConnectionNet1(Component *c1, Component *c2, double mass,
			       double freq, double loss, double collisionExponent,
			       double rattlingDistance, double x1, double x2)
    : Connection(c1, c2)
{
    GlobalSettings *gs = GlobalSettings::getInstance();

    energy = NULL;
    if (gs->getEnergyOn()) {
	energy = new double[gs->getNumTimesteps()];
	loss = 0.0;
    }

    string name1 = "nowhere", name2 = "nowhere";
    if (c1) name1 = c1->getName();
    if (c2) name2 = c2->getName();
    logMessage(1, "Creating connection between %s and %s", name1.c_str(), 
	       name2.c_str());
    this->mass = mass;
    this->freq = freq;
    this->loss = loss;
    this->collisionExponent = collisionExponent;
    this->rattlingDistance = rattlingDistance;
    this->x1 = x1;
    this->x2 = x2;

    itnum = 20;

    // get indices into component states
    i1 = -1;
    i2 = -1;
    if (c1) {
	//i1 = c1->getIndexf(x1);
	i1 = (int)floor(x1 * ((double)(c1->getStateSize()+1)));
	if (i1 > 0) i1--;
    }
    if (c2) {
	//i2 = c2->getIndexf(x2);
	i2 = (int)floor(x2 * ((double)(c2->getStateSize()+1)));
	if (i2 > 0) i2--;
    }

    logMessage(1, "Indices are %d, %d", i1, i2);

    // be consistent - if we only have one component, always make it c1
    if (c1 == NULL) {
	c1 = c2;
	i1 = i2;
	c2 = NULL;
	i2 = -1;
    }

    // initialise "state"
    u = 0.0;
    u1 = 0.0;
    u2 = 0.0;

    if (energy) {
	u1 = 0.0005;
	u2 = 0.0005;
    }

    double k = gs->getK();

    V = pow(2.0 * mass * loss, collisionExponent) / (2.0*k);
    Q = pow(mass * freq * freq, collisionExponent) / (collisionExponent+1.0);

    Gc = -(k*k)/mass;
    Gs1 = c1->getAlpha();

    logMessage(1, "V=%f,Q=%f,Gc=%g,Gs1=%g", V, Q, Gc, Gs1);

    if (c2 == NULL) {
	Wc[0] = -(Gc - Gs1) * V;
	Mc[0] = -(Gc - Gs1) * Q;
	logMessage(1, "Wc=%f,Mc=%f", Wc[0], Mc[0]);
    }
    else {
	Gs2 = c2->getAlpha();
	Wc[0] = -(Gc - Gs1) * V;
	Wc[1] = -(Gc) * V;
	Wc[2] = -(Gc) * V;
	Wc[3] = -(Gc - Gs2) * V;
	Mc[0] = -(Gc - Gs1) * Q;
	Mc[1] = -(Gc) * Q;
	Mc[2] = -(Gc) * Q;
	Mc[3] = -(Gc - Gs2) * Q;
	logMessage(1, "Gs2=%g,Wc=%f,%f,%f,%f,Mc=%f,%f,%f,%f", Gs2, Wc[0], Wc[1],
		   Wc[2], Wc[3], Mc[0], Mc[1], Mc[2], Mc[3]);
    }
}

ConnectionNet1::~ConnectionNet1()
{
    if (energy) delete[] energy;
}

void ConnectionNet1::runTimestep(int n)
{
    int qq;

    double k = GlobalSettings::getInstance()->getK();

    // In my tests, 1e-9 and 1e-12 both work fine, but 1e-9 only gives a very small
    // runtime gain over 1e-12, so let's use 1e-12 to be on the safe side
    double tol = 1e-12;
    //double qratol = 1e-12;

    if (energy) itnum = 50;

    // do basic state update first
    u = 2.0*u1 - u2;

    if (c2 == NULL) {
	//TaylorSeries ts;

	// single component connection
	// everything is scalar
	// b = -Ic*(u-u2);
	double b = (c1->getU()[i1] - c1->getU2()[i1]) - (u - u2);
	double a = u2 - c1->getU2()[i1];
	double r0 = -b;
	double qa = fabs(a) - rattlingDistance;
	qa = 0.5 * (qa + fabs(qa));
	
	//double qa_alphac1 = pow(qa, collisionExponent - 1.0);
	double qa_alphac1 = fastPow(qa, collisionExponent - 1.0);
	double qa_alphac = qa_alphac1 * qa;
	double phia = qa_alphac * qa;
	double sa = a >= 0.0 ? 1.0 : -1.0;
	double f1 = (collisionExponent + 1.0) * sa * qa_alphac;
	double f2 = (collisionExponent + 1.0) * collisionExponent * qa_alphac1;

	double q1 = u1 - c1->getU1()[i1];
	q1 = fabs(q1) - rattlingDistance;
	q1 = 0.5 * (q1 + fabs(q1));
	double sq1 = q1 >= 0.0 ? 1.0 : -1.0;
	//double Z = sq1 * pow(q1, collisionExponent - 1.0);
	double Z = sq1 * fastPow(q1, collisionExponent - 1.0);

	// O = II + Wc*Z;   where II is an identity matrix
	double O = 1.0 + Wc[0] * Z;
	double R;
	double r0last; // = 1.0;
	//double qralast = 1.0;
	
	for (qq = 1; qq <= itnum; qq++) {
	    double qra = fabs(r0 + a) - rattlingDistance;
	    qra = 0.5 * (qra + fabs(qra));

	    /*double pqra;
	    if (ts.valid) {
		pqra = ts.calculate(qra);
	    }
	    else {
		if (fabs((qra - qralast) / qralast) < qratol) {
		    ts.initialise(qra, collisionExponent);
		    pqra = ts.calculate(qra);
		}
		else {
		    pqra = pow(qra, collisionExponent);
		}
		qralast = qra;
		}*/

	    double pqra = pow(qra, collisionExponent);
	    double phira = pqra * qra;
	    double r0inv = 1.0 / max(fabs(r0), EPS);
	    double phiradiff = phira - phia;
	    double Rder;
	    if (fabs(r0) >= 1e-14) {
		double sr0 = r0 >= 0.0 ? 1.0 : -1.0;
		R = phiradiff * r0inv * sr0;
		double sr0a = (r0+a) >= 0.0 ? 1.0 : -1.0;
		Rder = ((collisionExponent+1.0) * sr0a * r0 * pqra - phiradiff) * r0inv * r0inv;
	    }
	    else {
		R = f1;
		Rder = f2;
	    }
	    double F = O*r0 + b + Mc[0]*R;
	    double L = Rder;
	    double Fder = O + Mc[0]*L;
	    r0 = r0 - F / Fder;

	    if ((!energy) && (qq > 1) && (r0last != 0.0)) {
		if (fabs((r0 - r0last) / r0last) < tol) break;
	    }
	    r0last = r0;
	}

	// update state with results
	u += Gc*Q*R + Gc*V*Z*r0;
	c1->getU()[i1] += Gs1*Q*R + Gs1*V*Z*r0;
    }
    else {
	// double component connection
	// everything is 2 element vector (except the 2x2 matrices)
	double b[2], a[2], r0[2], qa[2], qa_alphac1[2], qa_alphac[2];
	double phia[2], sa[2], f1[2], f2[2], q1[2], sq1[2], Z[2], O[4], R[2];
	double qra[2], pqra[2], phira[2], r0inv[2], phiradiff[2], Rder[2];
	double sr0[2], sr0a[2], F[2], Fder[4], result[2];
	double r0last[2];

	bool done1 = false, done2 = false;

	//r0last[0] = 1.0;
	//r0last[1] = 1.0;

	b[0] = (c1->getU()[i1] - c1->getU2()[i1]) - (u - u2);
	b[1] = (c2->getU()[i2] - c2->getU2()[i2]) - (u - u2);

	a[0] = u2 - c1->getU2()[i1];
	a[1] = u2 - c2->getU2()[i2];
	
	r0[0] = -b[0];
	r0[1] = -b[1];

	qa[0] = fabs(a[0]) - rattlingDistance;
	qa[0] = 0.5 * (qa[0] + fabs(qa[0]));
	qa[1] = fabs(a[1]) - rattlingDistance;
	qa[1] = 0.5 * (qa[1] + fabs(qa[1]));
	
	//qa_alphac1[0] = pow(qa[0], collisionExponent - 1.0);
	//qa_alphac1[1] = pow(qa[1], collisionExponent - 1.0);
	qa_alphac1[0] = fastPow(qa[0], collisionExponent - 1.0);
	qa_alphac1[1] = fastPow(qa[1], collisionExponent - 1.0);

	qa_alphac[0] = qa_alphac1[0] * qa[0];
	qa_alphac[1] = qa_alphac1[1] * qa[1];

	phia[0] = qa_alphac[0] * qa[0];
	phia[1] = qa_alphac[1] * qa[1];

	sa[0] = a[0] >= 0.0 ? 1.0 : -1.0;
	sa[1] = a[1] >= 0.0 ? 1.0 : -1.0;

	f1[0] = (collisionExponent + 1.0) * sa[0] * qa_alphac[0];
	f1[1] = (collisionExponent + 1.0) * sa[1] * qa_alphac[1];

	f2[0] = (collisionExponent + 1.0) * collisionExponent * qa_alphac1[0];
	f2[1] = (collisionExponent + 1.0) * collisionExponent * qa_alphac1[1];

	q1[0] = u1 - c1->getU1()[i1];
	q1[1] = u1 - c2->getU1()[i2];
	q1[0] = fabs(q1[0]) - rattlingDistance;
	q1[0] = 0.5 * (q1[0] + fabs(q1[0]));
	q1[1] = fabs(q1[1]) - rattlingDistance;
	q1[1] = 0.5 * (q1[1] + fabs(q1[1]));

	sq1[0] = q1[0] >= 0.0 ? 1.0 : -1.0;
	sq1[1] = q1[1] >= 0.0 ? 1.0 : -1.0;

	//Z[0] = sq1[0] * pow(q1[0], collisionExponent - 1.0);
	//Z[1] = sq1[1] * pow(q1[1], collisionExponent - 1.0);
	Z[0] = sq1[0] * fastPow(q1[0], collisionExponent - 1.0);
	Z[1] = sq1[1] * fastPow(q1[1], collisionExponent - 1.0);

	// O = II + Wc*Z;   where II is an identity matrix
	// row-major order
	O[0] = 1.0 + Wc[0] * Z[0];
	O[1] = Wc[1] * Z[1];
	O[2] = Wc[2] * Z[0];
	O[3] = 1.0 + Wc[3] * Z[1];

	for (qq = 1; qq <= itnum; qq++) {
	    qra[0] = fabs(r0[0] + a[0]) - rattlingDistance;
	    qra[1] = fabs(r0[1] + a[1]) - rattlingDistance;
	    qra[0] = 0.5 * (qra[0] + fabs(qra[0]));
	    qra[1] = 0.5 * (qra[1] + fabs(qra[1]));

	    //pqra[0] = pow(qra[0], collisionExponent);
	    //pqra[1] = pow(qra[1], collisionExponent);
	    pqra[0] = fastPow(qra[0], collisionExponent);
	    pqra[1] = fastPow(qra[1], collisionExponent);

	    phira[0] = pqra[0] * qra[0];
	    phira[1] = pqra[1] * qra[1];

	    r0inv[0] = 1.0 / max(fabs(r0[0]), EPS);
	    r0inv[1] = 1.0 / max(fabs(r0[1]), EPS);

	    phiradiff[0] = phira[0] - phia[0];
	    phiradiff[1] = phira[1] - phia[1];

	    if (fabs(r0[0]) >= 1e-14) {
		sr0[0] = r0[0] >= 0.0 ? 1.0 : -1.0;
		R[0] = phiradiff[0] * r0inv[0] * sr0[0];
		sr0a[0] = (r0[0]+a[0]) >= 0.0 ? 1.0 : -1.0;
		Rder[0] = ((collisionExponent+1.0) * sr0a[0] * r0[0] * pqra[0] - phiradiff[0]) * r0inv[0] * r0inv[0];
	    }
	    else {
		R[0] = f1[0];
		Rder[0] = f2[0];
	    }

	    if (fabs(r0[1]) >= 1e-14) {
		sr0[1] = r0[1] >= 0.0 ? 1.0 : -1.0;
		R[1] = phiradiff[1] * r0inv[1] * sr0[1];
		sr0a[1] = (r0[1]+a[1]) >= 0.0 ? 1.0 : -1.0;
		Rder[1] = ((collisionExponent+1.0) * sr0a[1] * r0[1] * pqra[1] - phiradiff[1]) * r0inv[1] * r0inv[1];
	    }
	    else {
		R[1] = f1[1];
		Rder[1] = f2[1];
	    }

	    F[0] = O[0]*r0[0] + O[1]*r0[1] + b[0] + Mc[0]*R[0] + Mc[1]*R[1];
	    F[1] = O[2]*r0[0] + O[3]*r0[1] + b[1] + Mc[2]*R[0] + Mc[3]*R[1];

	    Fder[0] = 1.0 + Mc[0]*Rder[0];
	    Fder[1] = Mc[1] * Rder[1];
	    Fder[2] = Mc[2] * Rder[0];
	    Fder[3] = 1.0 + Mc[3]*Rder[1];

	    // solve 2x2 linear system
	    double inve = 1.0 / Fder[3];
	    result[0] = (F[0] - ((Fder[1]*F[1])*inve)) /
		(Fder[0] - ((Fder[1]*Fder[2])*inve));
	    result[1] = (F[1] - (Fder[2]*result[0])) * inve;

	    //r0 = r0 - F / Fder;
	    r0[0] = r0[0] - result[0];
	    r0[1] = r0[1] - result[1];

	    if (qq > 1) {
		if ((!done1) && (r0last[0] != 0.0)) {
		    done1 = (fabs((r0[0]-r0last[0]) / r0last[0]) < tol);
		}
		if ((!done2) && (r0last[1] != 0.0)) {
		    done2 = (fabs((r0[1]-r0last[1]) / r0last[1]) < tol);
		}
	    }

	    if ((!energy) && (done1) && (done2)) break;
	    r0last[0] = r0[0];
	    r0last[1] = r0[1];
	}

	// update state
        // u = u+Gc*Qc*R+Gc*Vc*r0;
	u += Gc*Q*(R[0]+R[1]) + Gc*V*(Z[0]*r0[0]+Z[1]*r0[1]);

	c1->getU()[i1] += Gs1*Q*R[0] + Gs1*V*Z[0]*r0[0];
	c2->getU()[i2] += Gs2*Q*R[1] + Gs2*V*Z[1]*r0[1];

	/*u += Gc*Q*(R[0]+R[1]) + Gc*V*(r0[0]+r0[1]);

	c1->getU()[i1] += Gs1*Q*R[0] + Gs1*V*r0[0];
	c2->getU()[i2] += Gs2*Q*R[1] + Gs2*V*r0[1];*/
    }

    if (energy) {
	// connection's contribution to HL first
	double fe1 = ((0.5 * mass * u) / (k*k)) - ((0.5 * mass * u1) / (k*k));
	double fe2 = ((0.5 * mass * u1) / (k*k)) - ((0.5 * mass * u) / (k*k));
	double HL = (fe1*u) + (fe2*u1);

	// now compute HNL
	double HNL = 0.0;
	if (!c2) {
	    double Icu = u - c1->getU()[i1];
	    double qu = fabs(Icu) - rattlingDistance;
	    qu = 0.5 * (qu + fabs(qu));
	    double phiu = Q * pow(qu, collisionExponent + 1.0);
	    
	    double Icu1 = u1 - c1->getU1()[i1];
	    double qu1 = fabs(Icu1) - rattlingDistance;
	    qu1 = 0.5 * (qu1 + fabs(qu1));
	    double phiu1 = Q * pow(qu1, collisionExponent + 1.0);

	    HNL = 0.5 * (phiu + phiu1);
	}
	else {
	    double Icu[2];
	    double qu[2];
	    double phiu[2];
	    double Icu1[2];
	    double qu1[2];
	    double phiu1[2];

	    Icu[0] = u - c1->getU()[i1];
	    Icu[1] = u - c2->getU()[i2];
	    qu[0] = fabs(Icu[0]) - rattlingDistance;
	    qu[1] = fabs(Icu[1]) - rattlingDistance;
	    qu[0] = 0.5 * (qu[0] + fabs(qu[0]));
	    qu[1] = 0.5 * (qu[1] + fabs(qu[1]));
	    phiu[0] = Q * pow(qu[0], collisionExponent + 1.0);
	    phiu[1] = Q * pow(qu[1], collisionExponent + 1.0);
	    
	    Icu1[0] = u1 - c1->getU1()[i1];
	    Icu1[1] = u1 - c2->getU1()[i2];
	    qu1[0] = fabs(Icu1[0]) - rattlingDistance;
	    qu1[1] = fabs(Icu1[1]) - rattlingDistance;
	    qu1[0] = 0.5 * (qu1[0] + fabs(qu1[0]));
	    qu1[1] = 0.5 * (qu1[1] + fabs(qu1[1]));
	    phiu1[0] = Q * pow(qu1[0], collisionExponent + 1.0);
	    phiu1[1] = Q * pow(qu1[1], collisionExponent + 1.0);

	    HNL = 0.5 * (phiu[0] + phiu[1] + phiu1[0] + phiu1[1]);
	}

	energy[n] = HL + HNL;
    }

    // do state swap
    double tmp = u2;
    u2 = u1;
    u1 = u;
    u = tmp;
}

double *ConnectionNet1::getEnergy()
{
    return energy;
}

bool ConnectionNet1::coincides(ConnectionNet1 *conn)
{
    if ((c1 == conn->c1) && (c1 != NULL) && (i1 == conn->i1)) return true;
    if ((c2 == conn->c2) && (c2 != NULL) && (i2 == conn->i2)) return true;
    if ((c1 == conn->c2) && (c1 != NULL) && (i1 == conn->i2)) return true;
    return false;
}

void ConnectionNet1::getParallelTasks(vector<Task*> &tasks)
{
    // run this connection in parallel
    tasks.push_back(new TaskWholeConnection(this));
}

void ConnectionNet1::getSerialTasks(vector<Task*> &tasks)
{
    // nothing to do serially
}

