BACKEND_NAME="cuda" # cuda, x64
TAICHI_REPO="/home/taichigraphics/workspace/taichi2"

AOT_DIRECTORY="/tmp/aot_files"
RUNTIME_LIB="${TAICHI_REPO}/python/taichi/_lib/runtime"

rm -rf ${AOT_DIRECTORY}
mkdir -p ${AOT_DIRECTORY}

rm -rf build && mkdir build && cd build
cmake .. -DTAICHI_REPO=${TAICHI_REPO} && make -j && cd ..

python3 galaxy.py --dir=${AOT_DIRECTORY} --arch=${BACKEND_NAME}

echo "TI_LIB_DIR=${RUNTIME_LIB} ./build/galaxy ${AOT_DIRECTORY} ${BACKEND_NAME}"
TI_LIB_DIR=${RUNTIME_LIB} ./build/galaxy ${AOT_DIRECTORY} ${BACKEND_NAME}
