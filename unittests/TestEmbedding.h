/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Unit tests for Embedding
 */
#ifndef _TESTEMBEDDING_H_
#define _TESTEMBEDDING_H_

#include "cppunit/TestFixture.h"
#include "cppunit/TestSuite.h"
#include "cppunit/TestCaller.h"

#include "Embedding.h"

class TestEmbedding : public CppUnit::TestFixture {
 public:
    virtual void setUp();
    virtual void tearDown();

    void testEmbedding();

    static CppUnit::Test *suite() {
        CppUnit::TestSuite *suiteOfTests = new CppUnit::TestSuite("EmbeddingTest");
        suiteOfTests->addTest(new CppUnit::TestCaller<TestEmbedding>("testEmbedding", &TestEmbedding::testEmbedding));
        return suiteOfTests;
    }

};

#endif
