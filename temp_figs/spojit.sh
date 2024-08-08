seznam=`ls output/* | cut -d_ -f1,2,3 | sort |uniq`
for i in $seznam;
do
    echo $i
    convert $i* $i.pdf
done
