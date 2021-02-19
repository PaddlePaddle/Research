#!/bin/bash
main() {
    script_file=`readlink -f $0`
    bindir=`dirname $script_file`
    
    while [[ 1 -eq 1 ]];do
        sh -x $bindir/update_thread.sh 
        sleep 100
    done
}

main "$@"
