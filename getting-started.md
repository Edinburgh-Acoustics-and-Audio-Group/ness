# Getting started with the NESS Models

This guide presents a brief guide on using the NESS software to generate sounds.
You will need to use the Terminal (OS X or Linus) or Command Prompt (Windows). We are only doing very simple things with these programs though, so you if you don't have prior experience, you should still be able to follow along just fine.

### Video tutorial
You can see a video workthrough of this tutorial on the Acoustics and Audio Group YouTube channel:
[Link coming soon]()

## 1) Download and unzip the application
You can find the most recent release of the software here: [Ness code releases](https://github.com/Edinburgh-Acoustics-and-Audio-Group/ness/releases)

Download the zip file as relevant for your operating system, and unzip the file.
Inside the folder you'll find a few things. The key things to note here are:
 - **ness-framework**: this is the application itself. We will run it via command line in the steps below
 - **examples**: a folder of example scores and instruments that are helpful for understanding the parameters and the syntax the models understand
 - **doc**: a folder with more complete documentation of each model and the parameters and syntax required.

## 2) Open Terminal or the Command Prompt and navigate to the downloaded folder
 - On OS X or Linux, launch the Terminal.
 - On Windows, launch the Command Prompt. IMPORTANT: rather than opening this directly, you will need to select "Run as administrator"

Use the `cd` command to navigate to wherever you downloaded and unzipped the ness folder, e.g.
```bash
cd ~/Downloads/ness-osx-binary
```
or
```cmd
cd C:\Users\username\Downloads\ness-windows-binary
```

You can check you're in the right place by typing either `ls` (OSX/Linux) or 'dir' (Windows) to see if it shows the files in the ness folder.

## 3) Authorise the application (OS X only)
On a Mac, you'll need to explicitly authorise the binary with the following line:
```bash
chmod 755 ness-framework
```

## 4) Run an example script
To run the program, we need to specify two things
 - an *instrument file* (-i): this determines the nature of the instrument: often things like size, shape, material, etc. Consult the `doc` folder for more information.
 - a *score file* (-s): this describes how the available input parameters change over time: e.g. breath pressure, plucks, bow force, finger positions, etc. Again, see the `doc` folder for more information on any given model.

We can run the program with these inputs as follows:

#### OSX/Linux:
```bash
./ness-framework -i examples/instrument_brass.m -s examples/score_brass.m
```

#### Windows
```cmd
ness-framework -i examples/instrument_brass.m -s examples/score_brass.m
```

This should pause for a short time while the audio file is generated, and after 5-10 seconds you should see a new `output.wav` file in the ness folder.

## 5) Run a different example
The above step uses an example brass instrument and score file from the `examples` folder. There are examples for each of the available ness models:
 - bowed string
 - brass
 - guitar
 - modal plate
 - multi-plate
 - net1 (interconnected strings)
 - soundboard (strings connected to a plate)
 - zpt1 (interconnected plates)

To use a different model example, just specify the paths to the relevant score and instrument file in the `examples` folder, e.g.

```bash
./ness-framework -i examples/instrument_guitar.m -s examples/score_guitar.m
```
or 
```bash
./ness-framework -i examples/instrument_soundboard.txt -s examples/score_soundboard.txt
```

(if using Windows, remember you don't need the "./" at the start of these lines)

Note that different models give different kinds of outputs. The brass model outputs a single mono audio file. The guitar typically outputs one audio file per string (although it can also be set to output at several points from each string), along with a stereo mix of the strings combined. The soundboard will give an output for any number of positions on the soundboard, for each connected string, and again a stereo mix.


## 6) Developing your own score and instrument files
You are free to make changes to the example scripts to start to develop your own instrument and score files. A simple workflow for this might be the following:
 - create a new folder next to the examples folder, e.g. `user_tests`
 - copy a score and instrument pair from the examples into that folder (so that we keep intact copies of the original working examples)
 - open the score and/or instrument file in a text editor (any text editor will work here)
 - make changes as necessary (perhaps consulting the `doc` folder), then save your updated files
 - run the newly modified files with the ness framework:
   - `./ness-framework -i user_tests/instrument_guitar.m -s user_tests/score_guitar.m`
 - listen to the output
 - go back and make changes to the score and instrument files, then save
 - Run exactly the same line in the Terminal / Command Prompt (you can do this by just pressing the "up" key)
 - and so on...
