![Alt text](src/logo.gif?raw=true)

Jellyfish
======

Jellyfish is a program to simulated and view NMR spectra of spin systems in the liquid state experiencing J-couplings.

Requirements
------------

Jellyfish requires:
- [python](http://python.org/download/) >= 2.7 or [python](http://python.org/download/) >= 3.4

And the following python packages are required[1]:
- [numpy](http://sourceforge.net/projects/numpy/files/NumPy/) >= 1.8.2
- [matplotlib](http://matplotlib.org/) >= 1.4.2
- [scipy](http://sourceforge.net/projects/scipy/files/scipy/) >= 0.14.1
- [PyQt4](http://www.riverbankcomputing.com/software/pyqt/download) >= 4.11.4
- [h5py](http://www.h5py.org/) >= 2.5.0 (for loading Matlab data)

On Ubuntu and Debian these packages can be installed using the package manager:
```
sudo apt-get install python python-numpy python-matplotlib python-scipy python-qt4 python-h5py
```

On Windows these packages can easily be installed by downloading [Anaconda](http://continuum.io/downloads).

[1]: The program might work on older versions, but they have not been tested.

Installation
------------

###Linux###

To install Jellyfish, copy the Jellyfish directory to your favourite location (/usr/local/, for example).
Jellyfish can then be run by executing 'python /InstallPath/Jellyfish.py'.
Aliases or symlinks can be used to create a shortcut to start the program.

###Windows###

To install Jellyfish, copy the Jellyfish directory to your favourite location (C:\Program Files\, for example).
Jellyfish can then be run by double clicking on the 'WindowsRun.bat' file.
Alternatively, you can execute the 'WindowsInstall.vbs' file from within the Jellyfish directory.
This creates shortcuts on your desktop and in the startmenu, which you can use to run Jellyfish.

Contributing
------------

1. Fork it
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

Creators
--------

**Wouter Franssen**

**Bas van Meerten**


Contact
-------
For question and suggestions mail to: ssnake@science.ru.nl

License
-------

This project is licensed under the GNU General Public License v3.0 - See [LICENSE.md](LICENSE.md) for details.
