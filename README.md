# NESS Code Framework

This code was developed at the University of Edinburgh as part of the 2012-2016 European Research Council funded [Next Generation Sound Synthesis project](https://www.ness.music.ed.ac.uk/). It is a physical modelling synthesis code written in C++ (with optional CUDA acceleration) that allows various musical instruments and acoustic spaces to be simulated.


## Binary Packages

There are binary packages available for Windows (x64), macOS (x64 and arm64) and Linux (x64), including GPU-enabled packages for Windows and Linux. To run the GPU binaries you will need to have CUDA installed, and to run the Linux binaries you will need to install `libxml2` via your distribution's package manager, or from source. The binary packages include documentation and example input files - please see the *Basic Usage* section below for information on how to run the executable.


## Licence

The source code is released under the MIT licence, allowing both commercial and non-commercial re-use and modification. Please see the `LICENSE` file for full details, and the `AUTHORS` file for a list of contributors to the code.


## Compiling

### Linux

Building the code requires `g++`, `make` and `libxml2` to be installed. For GPU acceleration, NVIDIA's CUDA toolkit is also required. To build the pure C++ version of the code, simply change to the `src` directory and type `make`. To build the GPU accelerated version, run `make -f Makefile.cuda`.

If you get errors related to `libxml2`, make sure that the `xml2-config` utility is in your `PATH` before running `make`. If the library is installed using a system package then this should already be in a suitable location, but if you installed it manually then you may have to add it to your `PATH`.

### Windows

A Visual Studio 2019 solution and project file is included in the `src` directory for building the code. It may be possible to adapt this for other Visual Studio versions, or to adapt the Linux `Makefile` in order to build using MinGW or Cygwin.

You will need to build `libxml2` before building the NESS executable. There is a repository [here](https://github.com/kiyolee/libxml2-win-build) containing Visual Studio build files for this library, and similar repositories by the same user for its dependencies. Make sure to select the x64 configuration before building them (unless you specifically want to build a 32-bit NESS binary).

Once you have built `libxml2`, you can open the `NESS.sln` file in Visual Studio. You will probably need to update the library and include paths in the project settings to match where `libxml2` is on your system. To do this, right click on `NESS` in the **Solution Explorer** pane, and select **Properties**. The include paths are under **C/C++, All Options, Additional Include Directories**, while the library paths are under **Linker, General, Additional Library Directories**. Make sure that the target configuration is set to x64 (again, unless you specifically want to build a 32-bit binary).

To build the executable, right click on **Solution 'NESS'** in the **Solution Explorer** and select **Build Solution**. If all goes well, this will build the `ness-framework.exe` in a `Debug` subdirectory.

To build on Windows with CUDA support, you first need to install the CUDA toolkit including its Visual Studio integration. Then follow the same instructions as above, but use `NESSGPU.sln` instead of `NESS.sln`.

### macOS

To build on macOS you will need the command line developer tools installed, and also `libxml2`, though this should be installed by default. Simply switch to the `src` directory and run `make -f Makefile.macosx` to build for Intel CPU, or `make -f Makefile.macosx_arm` to build for ARM CPU.

GPU acceleration is not currently supported on macOS, however it may be possible to get it working by modifying the Linux `Makefile.cuda`.


## Basic Usage

The code takes two input files: an *instrument file*, which describes a configuration of components (for example strings, bars, plates, membranes, fretboards), and a *score file*, which describes a series of excitations of those components (for example plucks, strikes, bow movements). It runs a physical modelling simulation of those inputs and produces audio files as output.

To run the code on a given instrument and score file:

`  ness-framework -i instrument.txt -s score.txt`

The NESS project created models for several different types of instruments - linear and non-linear plates, bass drum, brass, guitar, modal plate, bowed string, and bar and string network. These are all implemented in this code framework. Example instrument and score files are given in the `examples` directory - each of these have their own instrument and score file formats. Some are simple plain text and some are based on a subset of Matlab syntax. It is also possible to "mix and match" components from different instrument families by using the XML instrument and score format, which supports all of the code's functionality.

For more detailed information about specific instrument models, please consult the files in the `doc` directory.


## Command Line Options

The main `ness-framework` executable can take a large number of command line options:

- **`-v`**: Prints out the code name and version number.
- **`-i <filename>`**: Specifies the instrument filename. Default is `instrument.txt`.
- **`-s <filename>`**: Specifies the score filename. Default is `score.txt`.
- **`-r`**: Enables saving of raw output data into `.f64` files. Off by default.
- **`-e`**: Run first 1000 timesteps and print out an estimate of how long it would take to run complete simulation.
- **`-o <basename>`**: Base name for output files. Default is `output`, so by default the files will be named `output-mix.wav`, `output1.wav`, `output2.wav`, etc. If there is only one output, the number will be omitted.
- **`-c [stereo|all]`**: Whether to save individual channel WAV files, or just the stereo mix. Default is to save all.
- **`-energy`**: Enables the energy conservation check. If supported by the components in use, this will result in an `energy.txt` file being created, containing the total energy in the system at each timestep. Enabling the energy check will set the default loss mode to 0 (no loss); if this is overridden and loss is re-enabled, energy will not be conserved. Enabling **`-energy`** also disables inputs such as strikes and bows, so impulses should be used instead, otherwise the system will be all zeroes throughout the run.
- **`-impulse`**: Instead of initialising the component state arrays to zero, a single unit impulse will be placed in the centre of the domain. Useful for testing in conjunction with **`-energy`**. In instruments with multiple components it is sometimes useful to have an impulse in only one of them; see the section below on component-local settings for details of how to do this.
- **`-normalise_outs/normalize_outs`**:  Normalises the output channel volumes individually before writing the stereo mix file. This means the channels will sound equally loud in the mix, even if their levels were different originally.
- **`-symmetric`**: Enables solving a symmetric linear system for non-linear plates and membranes. May be faster for some instruments but should give the same result. Off by default.
- **`-iterinv <count>`**: Number of iterations to run for embedded plates when dealing with the airbox coupling. Default is 5 for plates and 8 for membranes.
- **`-pcg_tol <tolerance>`**: Tolerance for conjugate gradient solver used by non-linear plates and membranes. Larger tolerances may make the code run faster at the expense of accuracy. Default is 1e-6.
- **`-pcg_max_it <count>`**: Maximum number of iterations for conjugate gradient solver used by non-linear plates and membranes. Default is 500.
- **`-no_recalc_q`**: Stops airbox initialisation from recalculating the Q parameter.
- **`-loss_mode <value>`**: Controls whether the components are lossy or not. 0 means sig0 and sig1 are both set to zero (no loss). -1 means sig0 is set to zero but sig1 is not. Default value is 1, which is normal loss.
- **`-fixpar <value>`**: Provides a scaling factor that is applied to the grid spacings for 1D and 2D components. Default is 1.0.
- **`-linear`**: Uses the faster linear code for plates and membranes. Off by default.
- **`-interpolate_inputs`**: Linearly interpolate inputs. Not implemented for bows.
- **`-interpolate_outputs`**: Linearly interpolate outputs.
- **`-negate_inputs`**: Negates all input values before applying them. Not implemented for bows.
- **`-log_state`**: Saves out raw component state arrays every timestep, for debugging. Can produce very large files, especially when there are 3D components present.
- **`-log_matrices`**: Saves out components' matrices after initialisation, for debugging. Not fully implemented.
- **`-disable_gpu`**: Disables use of the GPU, even when there is one available and the code has been built to use it.
- **`-max_threads <count>`**: Maximum number of threads to use. Default is number of CPU cores in the system.
- **`-cuda_2d_block_w, cuda_2d_block_h, cuda_3d_block_w, cuda_3d_block_h, cuda_3d_block_d`**: Sets block dimensions for GPU kernels. Default is 16x16 blocks for 2D and 8x8x8 for 3D.


In addition to these global settings, some settings can also be changed for a single component. This is done by putting a colon and then a component name after the setting name. For example:

- **`-iterinv:plate1 8`**
- **`-linear:drumbottom`**
- **`-interpolate_outputs:airbox`**
- **`-pcg_max_it:plate3 350`**

Also, multiple components can be specified in a comma separated list:

- **`-iterinv:plate1,plate2,plate4 8`**
- **`-linear:drumbottom,drumtop`**

Some settings (such as **`-max_threads`**) only make sense in a global context, so any component-specific versions of those will be ignored. The settings that affect inputs and outputs will affect all the inputs and outputs attached to the specified component.


## Logging

By default, the code prints a limited amount of information to the standard error stream. This can be made more verbose by setting the `NESS_LOG_LEVEL` environment variable: the default value is 5, but setting it to 3 will print more information, and 1 will produce very detailed information, sometimes useful for debugging.

The log messages can also be redirected to a file instead of stderr. To do this, set the `NESS_LOG_FILE` environment variable to the desired log file name.


## Tests

There is a suite of system tests and a suite of unit tests included with the code. The system tests run the entire code on various small test cases and check that the output is as expected. To run them:

- Ensure that the main executable has been built.
- Change to the `systemtests` directory.
- Run `make` to build the `compare` utility required by the tests (unless already built).
- Run `run.sh` to run the tests.

The unit tests exercise individual components within the code and check that they work as intended. These tests require `libcppunit` to be installed. To run them:

- Ensure that the main executable has been built. If you want to run the tests with GPU support, you will need to have built the main executable with GPU support.
- Change to the `unittests` directory.
- Run `make` to build the tests without GPU support, or `make -f Makefile.cuda` to build them with GPU support.
- Run `./main` to run the tests.

Please note that the unit tests and system tests have been developed solely under Linux and may not compile correctly on other platforms.
