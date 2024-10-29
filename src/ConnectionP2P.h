/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Represents a connection from one single point on a component to a single point on another
 * component (as opposed to, for example, an embedding that encompasses the whole component)
 */
#ifndef _CONNECTIONP2P_H_
#define _CONNECTIONP2P_H_

#include "Connection.h"

class ConnectionP2P : public Connection {
 public:
    ConnectionP2P(Component *c1, Component *c2, double xi1, double yi1, double zi1,
		  double xi2, double yi2, double zi2);
    virtual ~ConnectionP2P();

    virtual void runTimestep(int n) = 0;

    bool coincides(ConnectionP2P *conn);

 protected:
    double x1, y1, z1;
    double x2, y2, z2;

    int loc1, loc2;
};

#endif
