for dir in damping_hilbert damping_peaks
do	   

    stromy=`ls $dir/damping_*BK*png | grep "BK.*_" -o | sort | uniq`

    for i in $stromy
    do
	echo $i
	montage $dir/damping*$i*png -tile 4x4 -geometry +0+0 $dir/${i}_damping_all.png
	montage $dir/oscillation*$i*png -tile 4x4 -geometry +0+0 $dir/${i}_oscillation_all.png
    done

    convert $dir/*_damping_all.png $dir/damping_all.pdf
    convert $dir/*_oscillation_all.png $dir/oscillation_all.pdf

done
