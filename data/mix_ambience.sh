#!/bin/sh
FMT="-e signed-integer -t raw -b 16 -r 16000"
rm -rf 0 1 #2> /dev/null
mkdir 0
mkdir 1
# mix in ambience to "other" class
for f in other/rec*.raw
do
    fname=`basename $f`
    # move every 5th file to validation set
    python -c "import random; exit(1 if int(random.random()*5)==3 else 0)"
    if [ $? -eq 1 ]
    then
        cp $f 0/val$fname
    else
        # copy original sample
        cp $f 0/$fname
        # mix sample with abmience files
        for a in ambience/split*.raw
        do
            aname=`basename $a | sed -r 's/([^\.]+).*/\1/'`
            sox -m $FMT $f $FMT $a 0/$aname$fname
        done
    fi
done
# mix in ambience to vasilisa class
for f in vasilisa/rec*.raw
do
    fname=`basename $f`
    #move every 5th file to validation set
    python -c "import random; exit(1 if int(random.random()*5)==3 else 0)"
    if [ $? -eq 1 ]
    then
        cp $f 1/val$fname
    else
        # copy original sample
        cp $f 1/$fname
        # mix sample with abmience files
        for a in ambience/split*.raw
        do
            aname=`basename $a | sed -r 's/([^\.]+).*/\1/'`
            sox -m $FMT $f $FMT $a 1/$aname$fname
        done
    fi
done
#copy ambience samples to "other" class
cp ambience/*.raw 0

