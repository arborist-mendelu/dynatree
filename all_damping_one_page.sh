# damping_2021-03-22_BK01_M02.png

stromy=`ls damping/damping_*BK*png | grep "BK.*_" -o | sort | uniq`

for i in $stromy
do
    echo $i
    montage damping/damping*$i*png -tile 4x4 -geometry +0+0 damping/${i}_damping_all.png
    montage damping/oscillation*$i*png -tile 4x4 -geometry +0+0 damping/${i}_oscillation_all.png
done

convert damping/*_damping_all.png damping/damping_all.pdf
convert damping/*_oscillation_all.png damping/oscillation_all.pdf
