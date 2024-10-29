/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2012-2014. All rights reserved.
 *
 * The logging system.
 */
#ifndef _LOGGER_H_
#define _LOGGER_H_

// logs a message (using printf-style arguments), this may go to stderr or to a file
// depending on environment settings
void logMessage(int level, const char *msg, ...);

void logClose();

#endif
