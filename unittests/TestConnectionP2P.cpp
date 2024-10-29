/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 */

#include "TestConnectionP2P.h"
#include "DummyConnectionP2P.h"
#include "DummyComponent1D.h"

void TestConnectionP2P::setUp()
{
}

void TestConnectionP2P::tearDown()
{
}

void TestConnectionP2P::testConnectionP2P()
{
    Component *c1 = new DummyComponent1D("c1", 10);
    Component *c2 = new DummyComponent1D("c2", 10);
    Component *c3 = new DummyComponent1D("c3", 10);
    Component *c4 = new DummyComponent1D("c4", 10);

    DummyConnectionP2P *conn1 = new DummyConnectionP2P(c1, c2, 0.1, 0.1);
    DummyConnectionP2P *conn2 = new DummyConnectionP2P(c1, c3, 0.1, 0.1);
    DummyConnectionP2P *conn3 = new DummyConnectionP2P(c1, c4, 0.8, 0.1);
    DummyConnectionP2P *conn4 = new DummyConnectionP2P(c4, c1, 0.8, 0.1);

    CPPUNIT_ASSERT(conn1->coincides(conn2));
    CPPUNIT_ASSERT(!conn1->coincides(conn3));
    CPPUNIT_ASSERT(conn1->coincides(conn4));
    CPPUNIT_ASSERT(!conn2->coincides(conn3));
    CPPUNIT_ASSERT(conn2->coincides(conn4));
    CPPUNIT_ASSERT(!conn3->coincides(conn4));

    delete conn1;
    delete conn2;
    delete conn3;
    delete conn4;

    delete c1;
    delete c2;
    delete c3;
    delete c4;
}

