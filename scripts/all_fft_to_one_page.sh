stromy=`ls 0*/*fft/BK*png | grep "BK.*_" -o | sort | uniq`

for i in $stromy
  do montage 0*/*fft/$i*png -tile 4x4 -geometry +0+0 outputs/${i}fft_all.png
done

convert outputs/*fft_all.png outputs/fft.pdf
