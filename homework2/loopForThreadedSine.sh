
PROGRAM="./threadedSine"

OUTPUT_FILE="outputThreadedSine.txt"

> "$OUTPUT_FILE"

for i in {1..100}
do
    echo "Run #$i" >> "$OUTPUT_FILE"
    $PROGRAM >> "$OUTPUT_FILE" 2>&1  
    echo "--------------------" >> "$OUTPUT_FILE"
done

echo "Execution completed. Output saved in $OUTPUT_FILE."

