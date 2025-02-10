@REM set CONDA_ENV_DIR=D:\yina\dep-for-test
set CONDA_ENV_DIR=C:\Users\arda\miniforge3\envs\yina-npu\Lib\site-packages
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
cd ..