input=$1
output=$2

if [ -z "$output" ]; then
    ./singlethread < "$input"
elif [ "$output" = "none" ]; then
    ./singlethread < "$input" > /dev/null 2>&1
else
    ./singlethread < "$input" > "$output"
fi
