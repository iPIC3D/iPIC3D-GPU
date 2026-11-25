#!/bin/bash

outputFile="$(date +"%Y%m%d_%H%M%S")_summary.csv"


echo "Setup,Task Total[fields] (s),Task Total[particles] (s),Task Total[moments] (s),Simulation Time (s)" > "$outputFile"

for dir in */; do
    if [ "$dir" == "build/" ]; then
        continue
    fi

    logFile=$(find "$dir" -maxdepth 1 -type f -name '*.log')
    
    if [ -n "$logFile" ]; then
        # stage timers (TIME_TASKS): the "run max s" column, the last one, of
        # the regions of the field solves, the particles and the moments
        stageSum() {
            awk -v regions="$1" 'BEGIN { n = split(regions, r, " "); for (i = 1; i <= n; i++) want[r[i]] = 1 }
                 $1 == "Stage" && ($2 in want) { sum += $NF; found = 1 }
                 END { if (found) print sum }' "$logFile"
        }
        fieldsTime=$(stageSum "FieldSolveE FieldSolveB")
        particlesTime=$(stageSum "MoverLaunch ParticleExchange")
        momentsTime=$(stageSum "MomentsAwait")
        simulationTime=$(grep -E "Simulation Time *:" "$logFile" | sed -E 's/.*Simulation Time *: *([^ ]+) sec.*/\1/')

        fieldsTime=${fieldsTime:-"NA"}
        particlesTime=${particlesTime:-"NA"}
        momentsTime=${momentsTime:-"NA"}
        simulationTime=${simulationTime:-"NA"}

        echo "${dir%/},$fieldsTime,$particlesTime,$momentsTime,$simulationTime" >> "$outputFile"
    fi
done

echo "[*]Summary generated in $outputFile."
