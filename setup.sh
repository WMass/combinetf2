export COMBINETF2_BASE=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export PYTHONPATH="${COMBINETF2_BASE}:$PYTHONPATH"
export PATH="$PATH:${COMBINETF2_BASE}/bin"

echo "Created environment variable COMBINETF2_BASE=${COMBINETF2_BASE}"
