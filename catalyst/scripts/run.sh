#!/bin/bash

###################################################################################
#                                                                                 #
# VESTEC iPICmini launch script                                                   #
#                                                                                 #
# This script is intended to be run from the ParaView UI through a pvsc file      #
# When called by ParaView, this script takes 5 arguments:                         #
# $1 : SSH forwarding port                                                        #
# $2 : pvserver path                                                              #
# $3 : iPICmini path                                                              #
# $4 : input file                                                                 #
# $5 : number of MPI process                                                      #
#                                                                                 #
###################################################################################

client=$(echo $SSH_CONNECTION | cut -d ' ' -f1)

if [ -z $client ]; then
    client="localhost"
fi

echo "Script run from ${client}"

# Run pvserver
# @todo: test remote rendering without DISPLAY on OSMesa
DISPLAY=:0 $2 -rc --client-host=localhost -sp=$1 &
PIDPVSERVER=$!

# Run catalyst
mpirun -n $5 $3 $4 > /dev/null &
PIDCATALYST=$!

wait $PIDPVSERVER $PIDCATALYST
