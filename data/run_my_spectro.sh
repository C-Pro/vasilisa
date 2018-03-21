for f in  $1/*.raw
do
    cat $f | ./spectro $f.png
done
