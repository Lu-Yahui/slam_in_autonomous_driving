REPO=$(git rev-parse --show-toplevel)
pushd "$REPO/build"
cmake .. && make -j
popd
