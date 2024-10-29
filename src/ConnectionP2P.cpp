#include "ConnectionP2P.h"
#include "Logger.h"

ConnectionP2P::ConnectionP2P(Component *c1, Component *c2, double xi1, double yi1, double zi1,
			     double xi2, double yi2, double zi2) : Connection(c1, c2)
{
    x1 = xi1;
    x2 = xi2;
    y1 = yi1;
    y2 = yi2;
    z1 = zi1;
    z2 = zi2;

    loc1 = c1->getIndexf(x1, y1, z1);
    loc2 = c2->getIndexf(x2, y2, z2);
}

ConnectionP2P::~ConnectionP2P()
{
}

bool ConnectionP2P::coincides(ConnectionP2P *conn)
{
    if (((c1 == conn->c1) && (loc1 == conn->loc1)) ||
	((c2 == conn->c2) && (loc2 == conn->loc2)) ||
	((c1 == conn->c2) && (loc1 == conn->loc2)) ||
	((c2 == conn->c1) && (loc2 == conn->loc1))) {
	logMessage(1, "Found a co-incident connection");
	return true;
    }
    return false;
}
