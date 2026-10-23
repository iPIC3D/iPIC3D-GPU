#!/bin/bash


./clean.sh

echo "[*]Building iPIC3D"
./build.sh > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "[!]Build failed."
    exit 1
fi


for dir in */; do
    if [ "$dir" == "build/" ]; then
        continue
    fi

    inpFiles=( "$dir"*.inp )
    if [ ${#inpFiles[@]} -eq 1 ]; then

        cd "$dir"
        echo "[*]Benchmarking $dir"
        filename=$(date +"%Y%m%d_%H%M%S")
        logFile="${filename}.log"

        grid=$(echo "$dir" | cut -d'_' -f2)
        IFS='x' read -r -a dimensions <<< "$grid"
        product=1
        for dimension in "${dimensions[@]}"; do
            product=$((product * dimension))
        done
        export OMP_NUM_THREADS=2
        mpirun -np "$product" ../build/iPIC3D ./*.inp > "$logFile" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [!]Failed"
            exit 1
        fi

        cd ..

    elif [ ${#inpFiles[@]} -gt 1 ]; then
        echo "[!]Directory $dir contains multiple .inp files."
    fi
done

./outputCSV.sh
