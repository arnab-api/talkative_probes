SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
protoc --proto_path=${SCRIPT_DIR} --python_out=${SCRIPT_DIR} --pyi_out=${SCRIPT_DIR} ${SCRIPT_DIR}/patchscope.proto