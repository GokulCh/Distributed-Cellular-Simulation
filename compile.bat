@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

set MPI_PATH=C:\Program Files (x86)\Microsoft SDKs\MPI
set MPI_INC="%MPI_PATH%\Include"
set MPI_LIB="%MPI_PATH%\Lib\x64\msmpi.lib"

echo Compiling simulation.cpp...
cl /EHsc /O2 /I%MPI_INC% src\cpp\simulation.cpp /link %MPI_LIB% /out:simulation.exe

if %errorlevel% neq 0 (
    echo Compilation Failed!
    exit /b %errorlevel%
)

echo Compilation Success!
