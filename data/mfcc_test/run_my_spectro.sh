for f in  $1/*.raw
do
    cat $f | ./spec2 $f.png
done
