/*
 * NeSS Framework Code
 *
 * Copyright (c) The University of Edinburgh, 2014. All rights reserved.
 *
 * Manages settings for each individual component
 */
#ifndef _SETTINGS_MANAGER_H_
#define _SETTINGS_MANAGER_H_

#include <string>
#include <vector>
using namespace std;

struct Setting {
    string component;
    string setting;
    string value;
};

class SettingsManager {
 public:
    static SettingsManager *getInstance() {
	if (instance == NULL) instance = new SettingsManager();
	return instance;
    }

    // arg1 is the setting as on the command line, i.e. "-linear:plate1"
    // arg2 is the following argument if it exists and may (or may not) be a
    // parameter for the setting
    // returns true if the second argument got used
    bool putSetting(char *arg1, char *arg2 = NULL);

    // putSetting values for programmatic use
    // could call them all putSetting and rely on overloading, but may risk
    // automatic conversions between bool/int/double messing things up in
    // some cases so play it safe
    void putSetting(string compname, string settingname, string value);
    void putSetting(string compname, string settingname, double value);
    void putSettingInt(string compname, string settingname, int value);
    void putSettingBool(string compname, string settingname, bool value);

    string getStringSetting(string component, string setting);
    double getDoubleSetting(string component, string setting);
    int getIntSetting(string component, string setting);
    bool getBoolSetting(string component, string setting);

 private:
    static SettingsManager *instance;
    SettingsManager();
    ~SettingsManager();

    Setting *findSetting(string component, string setting);

    vector<Setting> settings;
};

#endif
