# Internals

This document is intended to give an overview of how the code works internally.

The most important concept in the framework is the component. Anything that the sound waves propagate through is a component. Components can be 1D, 2D or 3D. There is a hierarchy of classes to represent the different types:

```
Component - abstract superclass of all components. It maintains the state arrays for the
 |          current timestep (u) and two previous timesteps (u1, u2), contains all the inputs
 |          that affect this component, and performs a few other functions. The most
 |          important method is runTimestep, which actually updates the component for a
 |          single simulation timestep. This must be implemented by the subclasses. There
 |          is also the swapBuffers method that is called at the end of each timestep; by
 |          default it swaps the state array pointers ready for the next timestep.
 |
 +-Component1D - abstract superclass for all 1D finite difference components. Handles
 |  |            allocation of the 1D state arrays and computing locations within them.
 |  |
 |  +-ComponentString - models a simple linear string. Contains code for attaching the string
 |  |  |                to a soundboard.
 |  |  |
 |  |  +-StringWithFrets - models a string with frets behind it. Uses Newton's Method to
 |  |                      handle collisions between the string and the frets.
 |  |
 |  +-Bar - models a one dimensional bar with square cross section.
 |  |
 |  +-Fretboard - obsolete. Models a fretboard with a single string attached. Supports a
 |  |             moving finger as well as the standard pluck and strike inputs. Now
 |  |             superceded by GuitarString which supports the same features but is much
 |  |             more efficient.
 |  |
 |  +-BrassInstrument - models a brass instrument consisting of a tube with a number of
 |  |                   valves. The state arrays are a little different from most components
 |  |                   as brass requires separate pressure and velocity arrays, as well as
 |  |                   keeping more than two previous timesteps' data. However u, u1 and u2
 |  |                   still point to the most recent 3 timesteps of the main tube pressure
 |  |                   array so that standard Outputs work correctly. The brass update is
 |  |                   accelerated with AVX vector instructions, though this can be disabled
 |  |                   in the Makefile when building for machines without AVX.
 |  |
 |  +-GuitarString - models a single guitar string, with optional frets, backboard and
 |  |                movable fingers.
 |  |
 |  +-BowedString - models a single bowed string with optional bow(s) and movable fingers.
 |
 +-Component2D - abstract superclass for all 2D finite difference components. Handles
 |  |            allocation of the 2D state arrays and computing locations within them.
 |  |
 |  +-Plate - models a simple linear rectangular plate, as used in the Zero and ZeroPt1 codes.
 |  |  |
 |  |  +-SoundBoard - models the actual sound board used in the SoundBoard code. It is
 |  |                 a Zero-style Plate with a number of strings attached to it.
 |  |                 The string-board connections are handled in here instead of in a
 |  |                 separate Connection class as they are interdependent on each other.
 |  |
 |  +-PlateEmbedded - models a plate embedded within an airbox. Can handle both linear and
 |                    non-linear plates, rectangular or circular, and despite the name can
 |                    also model drum membranes. The actual interaction with the airbox is
 |                    handled in the Embedding class.
 |
 +-Component3D - abstract superclass for all 3D finite difference components. Handles
 |  |            allocation of the 3D state arrays and computing locations within them.
 |  |
 |  +-Airbox - abstract superclass for all airbox classes. This provides the interface and
 |     |       defines how an airbox should behave, but there can be different implementations
 |     |       with different capabilities and strengths.
 |     |
 |     +-AirboxIndexed - uses an index array so that it can model obstructions such as drum
 |                       shells within the airbox. Supports viscosity.
 |
 +-ModalPlate - modal (rather than finite difference) non-linear plate. This works differently
                from the finite difference components, so it inherits directly from Component.
```

Another important concept is the connection. A connection models some kind of interaction between two components. Again there is a hierarchy of classes:

```
Connection - abstract superclass for all connections.
 |
 +-ConnectionP2P - abstract superclass for all connections that link a single point on one
 |  |              component to a single point on another.
 |  |
 |  +-ConnectionZero - the simple point-to-point connection used in the Zero code.
 |  |
 |  +-ConnectionZeroPt1 - the simple point-to-point connection used in the ZeroPt1 code.
 |
 +-ConnectionNet1 - a connection between one or two bars or strings, used in the net1 code.
 |
 +-Embedding - models a PlateEmbedded within an Airbox.
```

Note that the connections between a soundboard and its strings are handled within the `SoundBoard` class; because they are potentially all dependent on each other, they don't fit the model of a `Connection` being between two components.


Inputs represent the stimuli from the score file. Each input is targetted at a single point on a particular component. The class hierarchy is:

```
Input - abstract superclass for all inputs. Manages component and location, stores start time
 |      and duration.
 |
 +-InputSample - superclass for all inputs that are taken from an audio sample in a buffer.
 |  |            The sample may either be read from a file or computed at startup.
 |  |
 |  +-InputStrike - simple sinusoidal strike on the component.
 |  |
 |  +-InputPluck - a half-sinusoid. Makes most sense for strings but can be used anywhere.
 |  |
 |  +-InputWav - an audio sample input read from a wav file.
 |
 +-InputBow - a bow. More complex than the other inputs, uses a Newton solver internally.
 |            This is only actually used to implement the simple bows used for the Zero code;
 |            the bow inputs for the bowed strings are implemented inside the BowedString class
 |            itself.
 |
 +-InputLips - a pair of lips blowing into an instrument. This only works for brass
 |             instruments.
 |
 +-InputValve - a valve that can open and close over time. Only works for brass instruments.
 |
 +-InputModalStrike - a strike input that works with the ModalPlate.
```

Some input types are so closely integrated with the components they affect that they do not have their own classes and are handled completely by the component classes. These include the fingers and bows used on guitar strings and bowed strings.


Outputs are similarly taken from a single point on a particular component. There are fewer types of output:

```
Output - superclass of all outputs. Not abstract, can be used when a simple default output is
 |       required.
 |
 +-OutputPressure - a pressure output. Instead of saving the value at each timestep, saves the
 |                  difference between current and previous (from two timesteps ago) values,
 |                  multiplied by the sample rate.
 |
 +-OutputDifference - similar to OutputPressure, but the value subtracted is from one timestep
 |                    ago instead of two.
 |
 +-OutputModal - an output that works with the ModalPlate.
```

The `Instrument` class ties all of this together. It contains a list of components, a list of outputs and a list of connections. Each component has an internal list of inputs that affect it and is responsible for updating them, there is no global list. `Instrument::runTimestep` calls the update methods on all the components, then all the connections. `Instrument::endTimestep` updates all the outputs and then swaps the component buffers. There is also a method that writes all the outputs to disk at the end. The `Instrument` class also manages a lot of the optimisation for GPU and multicore (see separate sections below).


There is no analogous `Score` class to encapsulate the data from the score file, as it makes more sense to associate the inputs directly with the components they affect.


The instrument file is read by classes in the `InstrumentParser` hierarchy. There is one for each format of input file supported. The top level parser will check the name of the executable and if it recognises it as one of the individual NESS code names ("zero", "zeroPt1", "mp3d", "soundboard", "guitar", etc.) it will use that code's specific parser. Otherwise it will try each parser in turn until one of them is able to read the file. The `ScoreParser` hierarchy works exactly the same way for parsing the score file (though the ZeroPt1 and MP3D score parsers actually just use the Zero parser as it does everything they need). There are also XML instrument and score parsers, along with several example instrument and score files. These are more flexible and general than the old-style plain text parsers and allow more freedom in how the different components are combined.


The modal plate component works quite differently from the other components and has a number of limitations. Because of the way the sample rate is determined when modal plates are in use, there can only be one modal plate in any simulation. Only special modal versions of the inputs and outputs (just `InputModalStrike` and `OutputModal`) will work with a modal plate, and connections and embeddings are not supported for modal plates at this time.


`main.cpp` manages the whole simulation lifecycle. It parses the command line, initialises everything, runs the main timestep loop, then shuts everything down cleanly at the end.


Other files:

- `BreakpointFunction` - implements a breakpoint function. Provides the same functionality as Matlab's `interp1` function but computes the values when they are needed instead of precomputing everything.
- `GlobalSettings` - a singleton class that manages settings that are used throughout the code, for example the sample rate and simulation duration.
- `Logger` - a simple logging system.
- `Material` - contains the properties of a material. Used by Zero and ZeroPt1.
- `MaterialsManager` - singleton class implementing a database of materials, indexed by name.
- `MathUtils` - contains generic mathematical functions (e.g. a Newton solver, linear interpolation, several dense matrix operations).
- `MatlabFunction` - an abstract superclass for functions that can be called from the Matlab format instrument and score files. There is a concrete subclass for each individual function.
- `MatlabParser` - a parser for a subset of Matlab files. Supports scalars, arrays, cell arrays and structures, including simple scalar expressions.
- `Parser` - both `InstrumentParser` and `ScoreParser` inherit from this. It provides some basic text file parsing utilities.
- `Profiler` - a simple profiling system.
- `SettingsManager` - manages a database of settings specific to each component, falling back to `GlobalSettings` for values not set locally.
- `Task` - defines a task that can potentially be executed in parallel with other tasks on different CPU cores. This is an abstract superclass, there are various subclasses for different task types. The most important is `TaskWholeComponent`, which encapsulates the entire state update for a single `Component` into a Task.
- `WavReader` - reads wav file inputs.
- `WavWriter` - writes wav file outputs.
- `WorkerThread` - wraps Pthreads or Windows threads for multicore acceleration (see below).
- `matgen` - generates update matrices for the `Plate` class.
- `matrix_unroll` - "unrolls" a sparse matrix into an index array and a set of co-efficients. Used by the GPU accelerated version of the `Plate` class.

The code within the `lib` subdirectory is mostly low level sparse linear algebra routines. It implements the CSR (compressed sparse row) matrices used extensively in the code, as well as optimised representations used in some components, and linear system solvers.

See GPU acceleration section below for details of the GPU-specific source files.


### Multicore Acceleration

This is done using Pthreads on Linux and native Win32 threads on Windows, as it gives us lower level control over when threads sleep than OpenMP. This is important because our timestep is so short that a significant amount of time is wasted if worker threads have to be woken up again every timestep. Most of the threading is implemented in the WorkerThread class, but at a higher level it is managed by the Instrument class.

At startup, the main code calls `Instrument::optimise`, which sets up the multithreading. This method scans through the entire list of components and calls `getSerialTasks` and `getParallelTasks` on each one, to build a complete list of parallel Tasks and a corresponding list of serial Tasks. From the multicore point of view, there are three different types of `Component`, and these are reflected in the Tasks that they add to the serial and parallel lists:

1. Components that return a single `TaskWholeComponent` to the parallel list, and nothing to the serial list. These are by far the most common. Their state updates can be run in parallel with those of other components, but there is no multithreading within the state update.
2. Components that return a single `TaskWholeComponent` to the serial list, but nothing to the parallel list. These are components whose state updates cannot be run concurrently with other components'.
3. Components that customise `getSerialTasks` and `getParallelTasks` to do something more complicated. So far, only the `ModalPlate` does this: it adds multiple tasks to the parallel list to divide up the work of a large matrix multiplication, and a single task to the serial list to tie everything together at the end.


Then a pool of worker threads is created. The number of worker threads is the maximum of the number of CPU cores in the system (determined by scanning `/proc/cpuinfo` on Linux, and through the Win32 API on Windows) and the number of tasks in the parallel list, unless overridden by a global setting. The worker threads spin until they are told to run the next timestep.

Then, in `Instrument::runTimestep`, the main thread tells the worker threads to run the next timestep, and then calls `WorkerThread::workerThreadRun` itself, because it also does a portion of the work itself. Each thread will start by running its own task, and will then loop, running more tasks as long as there are more needing run. Synchronisation is achieved by the GCC `__sync_*` functions and various volatile variables, again to ensure that the threads don't go to sleep. When all the tasks have been updated, the worker threads go back to spinning until it's time for them to run the next timestep, while the main thread runs the tasks that can't be run in parallel, then goes on with its other work (updating connections, taking outputs, etc.).

Only the first N tasks (where N is the number of threads) are actually "pinned" to a particular thread; the remainder are run by whichever thread gets to them first. In practice though, this will tend to be the same thread each time as each task usually takes about the same time to update from one timestep to the next. Also, in many cases there will be enough cores in the system for each task to get its own thread. There has been no effort to initialise tasks on the core they will ultimately run on ("first touch") or to parallelise anything other than the main component updates. This may be worth investigating if greater performance is needed.


Connections can now be parallelised in the same way as the component updates, though only `ConnectionNet1` takes advantage of this, being the only type that's both demanding enough to benefit from parallelisation, and able to be parallelised without causing problems. Parallel connection updates are run using the same worker threads as the components, but in a separate phase after all the component updates have completed.


### GPU Acceleration

Several component types (simple linear plates, non-linear modal plates and all types of airboxes) can be GPU accelerated using CUDA. All of the input and output types and point-to-point connections also have GPU implementations for efficiency when they are interfacing to a GPU-accelerated component. The `Embedding` is a special case; it can handle GPU-accelerated airboxes, but not GPU-accelerated plates.

Even on GPU systems, everything is initially set up the same as for non-GPU runs. The `Instrument::optimise` method, called after the input files have been parsed and the simulation objects created, deals with moving selected components and their dependencies to the GPU. Each component reports a GPU acceleration score: this is either "Great" (component works really well on GPU), "Good" (component can run on GPU and there is usually some benefit to doing so), or "No" (component cannot run on GPU). The optimiser iterates over all the components in the simulation, first attempting to move the ones that score "Great" to the GPU by calling `moveToGPU()` on them, and then does the same for the "Good" components. In most cases all the components that support GPU acceleration will be able to be moved successfully to the GPU. Components that are successfully moved to the GPU are moved onto a separate GPU components list, away from the serial and parallel lists described in the previous section.

Inputs, outputs and connections also need to be potentially moved to the GPU along with the components. Each component manages its own list of inputs, so it is responsible for moving them to the GPU if it gets moved to the GPU itself. The `Input` class also defines a `moveToGPU()` method for this purpose, which must be supported by all subclasses. Connections and outputs are managed separately by `Instrument`, so after moving components to the GPU, `Instrument::optimise` iterates over them all and calls `maybeMoveToGPU()` on each one. This gives them the chance to check whether the component(s) that they interact with have moved to the GPU and if so move themselves to the GPU as well.

For maximum flexibility, Connections should also be able to handle the case where one of the components they are connecting is running on the GPU but the other is not. This is likely to be unavoidably slow in a lot of cases, but allows for the simulation to still run no matter where each component has been placed.

Internally, there is a separate class for the GPU implementation of each object type, and the main object stores a pointer to an instance of this class. If this pointer is NULL, it means the object is not on the GPU. For example, the `Plate` class has a pointer to a `GPUPlate` object that handles the GPU acceleration for plates. Typically, the constructor for the GPU object deals with allocating GPU resources, copying any input data to them, etc.; the destructor will free these GPU resources; and there is a `runTimestep()` method that the main object's runTimestep() can delegate to when the object is GPU accelerated.

There is not a class hierarchy for the GPU objects. The GPU-specific modules are:

- `GPUAirboxIndexed` - GPU acceleration for indexed airboxes.
- `GPUConnectionZero` - GPU acceleration for Zero code-style connections.
- `GPUConnectionZeroPt1` - GPU acceleration for ZeroPt1 code-style connections.
- `GPUEmbedding` - GPU acceleration for embedding of plates within a GPU-accelerated airbox.
- `GPUInputBow` - GPU acceleration for bow input.
- `GPUInputModalStrike` - GPU acceleration for strike on modal plate.
- `GPUInputSample` - GPU acceleration for all descendents of InputSample. There is no need for separate classes to handle Strikes, Wavs, etc. as they are all treated the same after initialisation.
- `GPUModalPlate` - GPU acceleration for non-linear modal plate.
- `GPUOutput` - GPU acceleration for all types of finite difference output.
- `GPUOutputModal` - GPU acceleration for modal plate output.
- `GPUPlate` - GPU acceleration for linear Zero code-style plates.
- `GPUUtil` - provides useful CUDA utility functions

**WARNING**: when a Component has been moved to the GPU, the state array pointers returned by `getU()`, `getU1()` and `getU2()` will point to GPU memory space! All code that directly accesses a component's state arrays needs to be able to deal with this possibility.

