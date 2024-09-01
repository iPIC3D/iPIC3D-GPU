#!/bin/bash

outputFile="$(date +"%Y%m%d_%H%M%S")_summary.csv"


echo "Setup,Task Total[fields] (s),Task Total[particles] (s),Task Total[moments] (s),Simulation Time (s)" > "$outputFile"

for dir in */; do
    if [ "$dir" == "build/" ]; then
        continue
    fi

    logFile=$(find "$dir" -maxdepth 1 -type f -name '*.log')
    
    if [ -n "$logFile" ]; then
        fieldsTime=$(grep "Task Total\[fields\]:" "$logFile" | awk -F':' '{print $2}' | sed 's/s//;s/ //g')
        particlesTime=$(grep "Task Total\[particles\]:" "$logFile" | awk -F':' '{print $2}' | sed 's/s//;s/ //g')
        momentsTime=$(grep "Task Total\[moments\]:" "$logFile" | awk -F':' '{print $2}' | sed 's/s//;s/ //g')
        simulationTime=$(grep "Simulation Time:" "$logFile" | awk -F' ' '{print $3}')

        fieldsTime=${fieldsTime:-"NA"}
        particlesTime=${particlesTime:-"NA"}
        momentsTime=${momentsTime:-"NA"}
        simulationTime=${simulationTime:-"NA"}

        echo "${dir%/},$fieldsTime,$particlesTime,$momentsTime,$simulationTime" >> "$outputFile"
    fi
done

echo "[*]Summary generated in $outputFile."
