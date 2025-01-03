#!/bin/bash
export APPTAINER_BIND="/scratch,/cvmfs" 
if [[ -d $COMBINETF2_BASE ]]; then
    export APPTAINER_BIND="${APPTAINER_BIND},${COMBINETF2_BASE}/.."
fi
CONTAINER=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:v38

# Ensure kerberos permissions for eos access (requires systemd kerberos setup)
if [ -d $XDG_RUNTIME_DIR/krb5cc ]; then
    export KRB5CCNAME=/tmp/krb5cc_$UID 
    export APPTAINER_BIND="$APPTAINER_BIND,$XDG_RUNTIME_DIR/krb5cc:/tmp/krb5cc_$UID"
fi

singularity run $CONTAINER $@
