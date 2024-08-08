for base in $(ls *_0.png | sed 's/_0.png//'); do
    montage "${base}_0.png" "${base}_1.png" "${base}_2.png" "${base}_3.png" -tile 2x2 -geometry +0+0 "output/${base}.png"
done
